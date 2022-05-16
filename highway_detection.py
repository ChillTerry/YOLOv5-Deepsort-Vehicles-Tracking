import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox
import cv2 as cv
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from haversine import haversine, Unit
from sys import platform as _platform


class Detector:
    def __init__(self):
        """
        构造方法
        """
        self.img_size = 1280
        self.conf_thres = 0.5
        self.iou_thres=0.5

        # 加载模型
        self.weights = './weights/highway_m_300.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        
        # 开启半精度 直接可以加快运行速度 减少GPU占用 并且只有不明显的accuracy损失
        model.half()
        self.model = model

        # 标签名字列表['person', 'car', ...]
        self.name = model.module.names if hasattr(model, 'module') else model.names

    def preprocess(self, img):
        """
        图片预处理
        @param img list 输入图片
        @return img list 输出图片
        """
        # 深度学习模型输入图片的尺寸为正方形 而数据集中的图片一般为长方形
        # 粗暴的resize会使得图片失真 采用letterbox可以保持图片的长宽比例，剩下的部分采用灰色填充
        img = letterbox(img, new_shape=(self.img_size,self.img_size))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)

        # 模型改为半精度以后，输入也需要改成半精度
        img = img.half()

        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img


    def yolo_detect(self, img):
        """
        yolo目标检测
        @param img 输入图片
        @return pred_boxes np_array 输出识别物的位置和类别信息
        (690, 177, 706, 195, 'car', tensor(0.59482, device='cuda:0')
        """
        frame = self.preprocess(img)

        pred = self.model(frame, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                # 将经过letterbox处理的图片还原回来？
                det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], img.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.name[int(cls_id)]
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append((x1, y1, x2, y2, lbl, conf))
        return pred_boxes
            

class Tracker:
    """
    deepsort追踪
    """
    def __init__(self):
        """
        构造方法
        """
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

    def update_tracker(self, image, yolo_bboxes):
        """
        更新追踪结果
        """
        bbox_xywh = []
        confs = []
        clss = []

        for *x, cls_id, conf in yolo_bboxes:
            obj = [int((x[0]+x[2])/2), int((x[1]+x[3])/2), x[2]-x[0], x[3]-x[1]]
            bbox_xywh.append(obj)
            confs.append(conf)
            clss.append(cls_id)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        #更新追踪结果
        outputs = self.deepsort.update(xywhs, confss, clss, image)

        deepsort_bboxes = []
        for value in list(outputs):
            x1, y1, x2, y2, cls_, track_id = value
            deepsort_bboxes.append((x1, y1, x2, y2, cls_, track_id))

        return deepsort_bboxes


class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """
    def __init__(self, pixel_array, lonlat_array):
        assert pixel_array.shape==(4,2), "Need (4,2) input array"
        assert lonlat_array.shape==(4,2), "Need (4,2) input array"
        self.M = cv.getPerspectiveTransform(np.float32(pixel_array),np.float32(lonlat_array))
        self.invM = cv.getPerspectiveTransform(np.float32(lonlat_array),np.float32(pixel_array))
        
    def pixel_to_lonlat(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1,2)
        assert pixel.shape[1]==2, "Need (N,2) input array" 
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0],1))], axis=1)
        lonlat = np.dot(self.M,pixel.T)
        
        return (lonlat[:2,:]/lonlat[2,:]).T
    
    def lonlat_to_pixel(self, lonlat):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1,2)
        assert lonlat.shape[1]==2, "Need (N,2) input array" 
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0],1))], axis=1)
        pixel = np.dot(self.invM,lonlat.T)
        
        return (pixel[:2,:]/pixel[2,:]).T


class SpeedEstimate:
    def __init__(self):
        """
        配置相机画面与地图的映射点，需要根据自己镜头和地图上的点重新配置
        """
        quad_coords = {
            "lonlat": np.array([
                [30.221866, 120.287402], # top left
                [30.221527,120.287632], # top right
                [30.222098,120.285806], # bottom left
                [30.221805,120.285748] # bottom right
            ]),
            "pixel": np.array([
                [196,129],# top left
                [337,111], # top right
                [12,513], # bottom left
                [530,516] # bottom right
            ])
        }
        self.pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])

    def pixel2lonlat(self,x,y):
        """
        像素坐标转为经纬度
        """
        return self.pm.pixel_to_lonlat((x,y))[0]

    def pixelDistance(self,pa_x,pa_y,pb_x,pb_y):
        """
        相机画面两点在地图上实际的距离
        """
        lonlat_a = tuple(self.pixel2lonlat(pa_x,pa_y))
        lonlat_b = tuple(self.pixel2lonlat(pb_x,pb_y))

        return haversine(lonlat_a, lonlat_b, unit='m')


if __name__ == "__main__":
    img = cv.imread('./videos/snap.jpg')
    det = Detector()
    pred_boxes = det.yolo_detect(img)
    print(pred_boxes)


