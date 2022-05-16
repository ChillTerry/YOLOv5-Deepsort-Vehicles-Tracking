from highway_detection import Detector,Tracker,SpeedEstimate
import cv2

class Demo:

    def __init__(self):
        pass

    def plot_bboxes(self,image, bboxes):
        """
        绘制
        """

        for (l, t, r, b, cls_name, track_id) in bboxes:
            if cls_name == 'car':
                color = (0, 0, 255)
            elif cls_name == 'truck':
                color = (0, 255, 0)

            cv2.rectangle(image, (l,t), (r,b), color, thickness=2)

            cv2.putText(image, '{}-{}'.format(cls_name, track_id), (l,t-10), cv2.FONT_ITALIC,0.4,color,1)


    def detect(self):
        cap = cv2.VideoCapture('./videos/highway.mp4')
        # 实例化1个检测器
        yolo_detector = Detector()
        # 实例化1个追踪器
        tracker_l = Tracker()

        while True:
            ret,frame = cap.read()

            frame = cv2.resize(frame, (562,1000))

            # 调用检测
            yolo_bboxes = yolo_detector.yolo_detect(frame)

            # 调用追踪
            track_bboxes = tracker_l.update_tracker(frame, yolo_bboxes)
            
            # 绘制
            self.plot_bboxes(frame, track_bboxes)

            cv2.imshow('Demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

demo = Demo()
demo.detect()
            