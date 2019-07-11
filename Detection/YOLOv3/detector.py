import cv2
import sys
import torch
import random
import pickle as pkl
import os.path as osp
from datetime import datetime
from torch.autograd import Variable
from Detection.YOLOv3.darknet import Darknet
from Detection.YOLOv3.util import process_result, cv_image2tensor, transform_result

from Detection.VideoBasedObjectsDetectionAlgorithm import VideoBasedObjectsDetectionAlgorithm


class YOLOv3(VideoBasedObjectsDetectionAlgorithm):
    @staticmethod
    def draw_bbox(imgs, bbox, colors, classes, file):
        if int(bbox[-1]) < len(classes):
            img = imgs[int(bbox[0])]
            label = classes[int(bbox[-1])]
            p1 = tuple(bbox[1:3].int())
            p2 = tuple(bbox[3:5].int())
            YOLOv3.write_bbox_in_file(file, label, [int(p1[0]), int(p1[1]), int(p2[0] - p1[0]), int(p2[1] - p1[1])])
            color = random.choice(colors)
            cv2.rectangle(img, p1, p2, color, 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
            p3 = (p1[0], p1[1] - text_size[1] - 4)
            p4 = (p1[0] + text_size[0] + 4, p1[1])
            cv2.rectangle(img, p3, p4, color, -1)
            cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)

    @staticmethod
    def detect_video(model, args):
        input_size = [int(model.net_info['height']), int(model.net_info['width'])]

        colors = pkl.load(open("Detection/YOLOv3/pallete", "rb"))
        classes = YOLOv3.load_classes("Detection/YOLOv3/data/coco.names")
        colors = [colors[1]]

        output_path = osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + '.mp4')
        cap, out = YOLOv3.get_videocap_videowriter(args.input, output_path)

        read_frames = 0

        detected_objects = \
            open(osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + ".txt"), 'w')

        start_time = datetime.now()
        print('Detecting...')
        while cap.isOpened():
            retflag, frame = cap.read()
            read_frames += 1
            if retflag:
                frame_tensor = cv_image2tensor(frame, input_size).unsqueeze(0)
                frame_tensor = Variable(frame_tensor)

                if args.cuda:
                    frame_tensor = frame_tensor.cuda()

                detections = model(frame_tensor, args.cuda).cpu()
                detections = process_result(detections, 0.5, 0.4)
                detected_objects.write(str(read_frames - 1))
                if len(detections) != 0:
                    detections = transform_result(detections, [frame], input_size)
                    for detection in detections:
                        YOLOv3.draw_bbox([frame], detection, colors, classes, detected_objects)
                detected_objects.write("\n")

                if not args.no_show:
                    cv2.imshow('frame', frame)
                out.write(frame)
                if read_frames % 30 == 0:
                    print('Number of frames processed:', read_frames)
                if not args.no_show and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        end_time = datetime.now()
        print('Detection finished in %s' % (end_time - start_time))
        print('Total frames:', read_frames)
        cap.release()
        out.release()
        if not args.no_show:
            cv2.destroyAllWindows()

        print('Detected video saved to ' + output_path)

        return

    @staticmethod
    def run(args):
        if args.cuda and not torch.cuda.is_available():
            print("ERROR: cuda is not available, try running on CPU")
            sys.exit(1)

        print('Loading network...')
        model = Darknet('Detection/YOLOv3/cfg/yolov3.cfg')
        model.load_weights('Detection/YOLOv3/yolov3.weights')
        if args.cuda:
            model.cuda()

        model.eval()
        print('Network loaded')

        YOLOv3.detect_video(model, args)
