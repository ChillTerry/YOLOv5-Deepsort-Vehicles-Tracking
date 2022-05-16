import torch
from highway_detection import Detector,Tracker,SpeedEstimate
import cv2 as cv
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

class HighwayAnalyse:
    def __init__(self):
        self.yolo_detector = Detector()
        self.deepsort_tracker = Tracker()
        self.speed_estimate = SpeedEstimate()
        # 记录左右车道上次位置、速度 { id :{'last_pos':(123,234),'speed':12.34}  , 2: ........} 
        self.last_frame_info = {} 
        self.center_points = {}
        self.track_tail_points = {}

    def plot_info(self, image, deepsort_boxes):
        for (l, t, r, b, cls_name, track_id) in deepsort_boxes:
            # 速度
            if track_id in self.last_frame_info:
                if self.last_frame_info[track_id]['speed'] != 0:
                    scale = (r- l)/40 * 0.5
                    speed =  round(self.last_frame_info[track_id]['speed'], 1)
                    cv.putText(image, '{}km/h'.format(  speed ), (l-20,t-10), cv.FONT_ITALIC,scale,(0,255,0),1)
            
            center_pos = (l+r)/2, (t+b)/2
            if track_id in self.track_tail_points:
                # 拖着小尾巴
                self.track_tail_points[track_id].append(center_pos)
                if len(self.track_tail_points[track_id]) > 20:
                    del self.track_tail_points[track_id][0]
            else:   
                # 拖着小尾巴
                self.track_tail_points[track_id] = [center_pos]
            # print(self.track_tail_points)
            # 绘制尾部
            points = np.array(self.track_tail_points[track_id], np.int32)
            points = points.reshape((-1,1,2))
            # print(points)
            cv.polylines(image, [points], False, (255,255,0), 1)


    def update_id_info(self,deepsort_boxes,side='left'):
        """
        获取当前画面各ID的位置、车速信息
        """
        this_frame_info = {}
        for (l, t, r, b, cls_name, track_id) in deepsort_boxes:
            center_pos = ((l+r)/2, (t+b)/2)
            this_frame_info[track_id] = {'last_pos':center_pos, 'speed':0}
        # print('this_frame_info', this_frame_info)

        # if self.last_frame_info is not None:
        if len(self.last_frame_info) > 0:
            for key,val in this_frame_info.items():
                if key in self.last_frame_info:
                    this_frame_pos = val['last_pos']
                    last_frame_pos = self.last_frame_info[key]['last_pos']
                    distance = self.speed_estimate.pixelDistance(this_frame_pos[0],
                                this_frame_pos[1], last_frame_pos[0], last_frame_pos[1])
                    speed = distance * 3.6      
                    this_frame_info[key]['speed'] = speed   
                else:
                    this_frame_info[key] = {'last_pos':center_pos, 'speed':0}   
        self.last_frame_info = this_frame_info
        # print('last_frame_info', self.last_frame_info)

    def detect(self):
        cap = cv.VideoCapture('./videos/highway.mp4')
        fps = round(cap.get(cv.CAP_PROP_FPS))
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        video_writer = cv.VideoWriter('./output.mp4', fourcc, fps, (562,1000))
        # 前2帧可能未识别，过滤掉
        frame_index = -2

        while True:
            ret,frame = cap.read()
            # frame = cv.flip(frame, 1)
            frame = cv.resize(frame, (562,1000))

            yolo_boxes = self.yolo_detector.yolo_detect(frame)
            # print('yolo_boxes: ', yolo_boxes)
            deepsort_boxes = self.deepsort_tracker.update_tracker(frame, yolo_boxes)
            # print('deepsort_boxes: ', deepsort_boxes)

            if (frame_index % fps) == 0:
                # print('frame_index: ', frame_index)
                self.update_id_info(deepsort_boxes)
            self.plot_info(frame, deepsort_boxes)
            

            for *x, label_name, conf in yolo_boxes:
                cv.rectangle(frame, (x[0],x[1]), (x[2],x[3]), (0,255,0), 1)

            frame_index += 1
            cv.imshow('demo', frame)
            video_writer.write(frame)
            if cv.waitKey(10) & 0xff==27:
                break
        
        cap.release()
        video_writer.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    hightway = HighwayAnalyse()
    hightway.detect()