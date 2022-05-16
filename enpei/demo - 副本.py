from highway_detection import Detector,Tracker,SpeedEstimate
import cv2

class HighwayAnalyse:
    def __init__(self):

        self.speed_esti = SpeedEstimate()
        # 记录左右车道上次位置、速度 { id :{'last_pos':(123,234),'speed':12.34}  , 2: ........} 
        self.left_ids_info = {} 
        self.right_ids_info = {}

    def plot_bboxes(self,image, bboxes,side='left'):
        """
        绘制
        """
        
        this_ids_info = self.left_ids_info if side == 'left' else self.right_ids_info

        for (l, t, r, b, cls_name, track_id) in bboxes:
            if cls_name == 'car':
                color = (0, 0, 255)
            elif cls_name == 'truck':
                color = (0, 255, 0)

            cv2.rectangle(image, (l,t), (r,b), color, thickness=1)

            if side == 'left':
                # 尾部位置（左）
                head_pos = l+int((r-l)/2),t
            else:
                head_pos = l+int((r-l)/2),b


            cv2.circle(image, head_pos, 3, (0,0,255),-1)

            if track_id in this_ids_info and this_ids_info[track_id]['speed'] != 0:
                cv2.putText(image, '{} km/h'.format(  round(this_ids_info[track_id]['speed'],1)), (l,t-10), cv2.FONT_ITALIC,0.4,color,1)
            else:
                cv2.putText(image, '{}-{}'.format(cls_name, track_id), (l,t-10), cv2.FONT_ITALIC,0.4,color,1)



    def update_id_info(self,bboxes,side='left'):
        """
        获取当前画面各ID的位置、车速信息
        """
        this_ids_info = self.left_ids_info if side == 'left' else self.right_ids_info

        # 复制上帧位置
        last_frame_pos = {}
        for key,val in this_ids_info.items():
            last_frame_pos[key] = {'last_pos':val['last_pos']} 

        # 本帧位置
        this_frame_pos = {}
        for (l, t, r, b, cls_name, track_id) in bboxes:

            if side == 'left':
                # 尾部位置（左）
                head_pos = l+int((r-l)/2),t
            else:
                head_pos = l+int((r-l)/2),b
            # 初始化
            this_frame_pos[track_id] = {'last_pos':head_pos,'speed':0}

        # 如果两次都有位置，计算速度并返回
        new_ids_info = {}
        if len(last_frame_pos) > 0:
            # 遍历本次所有ID
            car_num = 0
            for key,val in this_frame_pos.items():
                # 取两帧同一ID
                if key in last_frame_pos:
                    # 获取本帧的位置
                    this_frame_val = val['last_pos']
                    # 获取上帧的位置
                    last_frame_val = last_frame_pos[key]['last_pos']
                    # 计算记录
                    distance = self.speed_esti.pixelDistance(this_frame_val[0],this_frame_val[1], last_frame_val[0],last_frame_val[1])
                    # 速度 km/h
                    speed = distance * 3.6 
                    # 重新赋值
                    new_ids_info[key] = {'last_pos':this_frame_val,'speed':speed}

                    car_num += 1

            print("{}侧车道{}辆车位置信息、速度已经更新，0.5秒后刷新".format(side,car_num))

        else:
            new_ids_info = this_frame_pos

            print("{}侧车道{}辆车位置信息更新，0.5秒后计算速度".format(side,len(new_ids_info)))
        # 重新赋值
        if side == 'left':
            self.left_ids_info = new_ids_info
        else:
            self.right_ids_info = new_ids_info


        
    def main(self):

        # 实例化1个检测器
        yolo_detector = Detector()

        # 实例化2个追踪器
        tracker_l = Tracker()
        tracker_r = Tracker()

        # 仅对该视频有效，其他视频需要重新训练、修改相应位置参数
        cap = cv2.VideoCapture('videos/highway.mp4')

        fps = round(cap.get(cv2.CAP_PROP_FPS))

        print("当前视频帧率为：{}".format(fps))

        # 前2帧可能未识别，过滤掉
        frame_index = -2

        while True:

            ret,frame  = cap.read()

            if frame is None:
                break

            # 这里对原视频进行了缩放，在SpeedEstimate配置中的坐标点也需要缩放
            frame = cv2.resize(frame, (562,1000))

            # 划线
            # cv2.line(frame,(281,0),(281,1000),(0,255,0),2)

            # 将视频划分为左右两车道
            crop_l =  frame[:,:281]
            crop_r =  frame[:,281:]
            
            # 调用YOLO检测接口

            yolo_bboxs_l = yolo_detector.yolo_detect(crop_l)
            yolo_bboxs_r = yolo_detector.yolo_detect(crop_r)

            # 调用追踪接口
            tracker_bboxs_l = tracker_l.update_tracker(crop_l,yolo_bboxs_l)
            tracker_bboxs_r = tracker_r.update_tracker(crop_r,yolo_bboxs_r)

            # 每隔 FPS 帧刷新一下位置(1s)，并计算速度
            if (frame_index % fps) == 0:
                self.update_id_info(tracker_bboxs_l,'left')
                self.update_id_info(tracker_bboxs_l,'right')

            #绘制
            self.plot_bboxes(crop_l, tracker_bboxs_l,'left')
            self.plot_bboxes(crop_r, tracker_bboxs_r,'right')

            frame[:,:281] = crop_l
            frame[:,281:] = crop_r

            frame_index +=1

            cv2.imshow('Demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



highway = HighwayAnalyse()
highway.main()

# speed_esti = SpeedEstimate()
# a = speed_esti.pixelDistance(17, 988, 637, 376)
# print(a)