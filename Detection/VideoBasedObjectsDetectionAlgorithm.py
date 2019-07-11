import sys
import cv2
import math
import torch


class VideoBasedObjectsDetectionAlgorithm:
    @staticmethod
    def load_classes(namesfile):
        fp = open(namesfile, "r")
        names = fp.read().split("\n")[:-1]
        return names

    @staticmethod
    def create_batches(imgs, batch_size):
        num_batches = math.ceil(len(imgs) // batch_size)
        batches = [imgs[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

        return batches

    # bbox is [x, y, w, h]
    @staticmethod
    def write_bbox_in_file(file, label, bbox):
        file.write(" ".join(["\t", label, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]))

    @staticmethod
    def draw_bbox(imgs, bbox, colors, classes, file):
        return

    @staticmethod
    def get_videocap_videowriter(inp_video_path, out_video_path):
        videocap = cv2.VideoCapture(inp_video_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = int(videocap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videocap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = videocap.get(cv2.CAP_PROP_FPS)

        videowriter = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        return videocap, videowriter

    @staticmethod
    def detect_video(model, args):
        return

    @staticmethod
    def run(args):
        if args.cuda and not torch.cuda.is_available():
            print("ERROR: cuda is not available, try running on CPU")
            sys.exit(1)

        # Load network
        # VideoBasedObjectsDetectionAlgorithm.detect_video(model, args)
