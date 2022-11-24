import cv2
import time
from pathlib import Path

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from sort import *

def run_inference_for_single_image(im_path):

    threshold = 0.25
    iou_threshold = 0.45

    with torch.no_grad():
        # Initialize
        set_logging()
        device = select_device('0')
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load("model-retail/model.pt", map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(640, s=stride)  # check img_size

        trace = True
        if trace:
            model = TracedModel(model, device, 640)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                device).eval()

        # Set Dataloader
        dataset = LoadImages(im_path, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        for i in range(len(names)):
            if names[i] == 'person':
                names[i] = 'persona'
            if names[i] == 'face':
                names[i] = 'rostro'
        print(names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            augment = False
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, threshold, iou_threshold, classes=None,
                                       agnostic=False)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, 'Clase\t\tCantidad\n', im0s, getattr(dataset, 'frame', 0)

                im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                p = Path(p)  # to Path
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{names[int(c)]}:\t{n}\n"  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                return im0, s

class Work(QThread):
    Imageupd = pyqtSignal(QImage)
    Labelupd = pyqtSignal(str)
    threshold = 0.25
    iou_threshold = 0.45

    def __init__(self, path):
        super().__init__()
        self.video_path = path
        self.running = True
        self.pause_video = False

    def run(self):
        try:
            with torch.no_grad():
                # Initialize
                set_logging()
                device = select_device('0')
                half = device.type != 'cpu'  # half precision only supported on CUDA

                # Load model
                model = attempt_load("model-retail/model.pt", map_location=device)  # load FP32 model
                stride = int(model.stride.max())  # model stride
                imgsz = check_img_size(640, s=stride)  # check img_size

                trace = True
                if trace:
                    model = TracedModel(model, device, 640)

                if half:
                    model.half()  # to FP16

                # Second-stage classifier
                classify = False
                if classify:
                    modelc = load_classifier(name='resnet101', n=2)  # initialize
                    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                        device).eval()

                # Set Dataloader
                vid_path, vid_writer = None, None
                dataset = LoadImages(self.video_path, img_size=imgsz, stride=stride)

                # Get names and colors
                names = model.module.names if hasattr(model, 'module') else model.names
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

                # Run inference
                if device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                old_img_w = old_img_h = imgsz
                old_img_b = 1

                t0 = time.time()
                for path, img, im0s, vid_cap in dataset:
                    augment = False
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Warmup
                    if device.type != 'cpu' and (
                            old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                        old_img_b = img.shape[0]
                        old_img_h = img.shape[2]
                        old_img_w = img.shape[3]
                        for i in range(3):
                            model(img, augment=augment)[0]

                    # Inference
                    t1 = time_synchronized()
                    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                        pred = model(img, augment=augment)[0]
                    t2 = time_synchronized()

                    # Apply NMS
                    pred = non_max_suppression(pred, self.threshold, self.iou_threshold, classes=None,
                                               agnostic=False)
                    t3 = time_synchronized()

                    # Apply Classifier
                    if classify:
                        pred = apply_classifier(pred, modelc, img, im0s)

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        p, s, im0, frame = path, 'Clase\tCantidad', im0s, getattr(dataset, 'frame', 0)
                        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                        p = Path(p)  # to Path
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{names[int(c)]}:\t{n}\n"  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                label = f'{names[int(cls)]}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                        # Print time (inference + NMS)
                        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                        convertir_QT = QImage(im0.data, im0.shape[1], im0.shape[0],
                                              QImage.Format_RGB888)
                        pic = convertir_QT.scaled(600, 480, Qt.KeepAspectRatioByExpanding)
                        self.Imageupd.emit(pic)
                        self.Labelupd.emit(s)

        except Exception as e:
            return None

    def stop(self):
        self.quit()
        self.running = False

    def pause(self):
        self.pause_video = True

    def play(self):
        self.pause_video = False

class Work_track(QThread):
    Imageupd = pyqtSignal(QImage)
    Labelupd = pyqtSignal(str)
    threshold = 0.25
    iou_threshold = 0.45
    thickness = False
    nobbox = False
    show_track = False

    def __init__(self, path):
        super().__init__()
        self.video_path = path
        self.running = True
        self.pause_video = False

    """Function to Draw Bounding boxes"""

    def draw_boxes(self, img, bbox, text_result, identities=None, categories=None, confidences=None, names=None, colors=None):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            tl = self.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            # conf = confidences[i] if confidences is not None else 0

            color = colors[cat]

            if not self.nobbox:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

            label = str(id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
            text_result += names[cat] + ":" + str(id) + "\n"
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        return img, text_result

    def run(self):
        try:
            sort_tracker = Sort(max_age=5,
                                min_hits=2,
                                iou_threshold=0.2)

            with torch.no_grad():
                # Initialize
                set_logging()
                device = select_device('0')
                half = device.type != 'cpu'  # half precision only supported on CUDA

                # Load model
                model = attempt_load("model-retail/model.pt", map_location=device)  # load FP32 model
                stride = int(model.stride.max())  # model stride
                imgsz = check_img_size(640, s=stride)  # check img_size

                trace = True
                if trace:
                    model = TracedModel(model, device, 640)

                if half:
                    model.half()  # to FP16

                # Second-stage classifier
                classify = False
                if classify:
                    modelc = load_classifier(name='resnet101', n=2)  # initialize
                    modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                        device).eval()

                # Set Dataloader
                vid_path, vid_writer = None, None
                dataset = LoadImages(self.video_path, img_size=imgsz, stride=stride)

                # Get names and colors
                names = model.module.names if hasattr(model, 'module') else model.names
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

                # Run inference
                if device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                old_img_w = old_img_h = imgsz
                old_img_b = 1

                t0 = time.time()
                startTime = 0

                for path, img, im0s, vid_cap in dataset:
                    augment = False
                    text_results = ''
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Warmup
                    if device.type != 'cpu' and (
                            old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                        old_img_b = img.shape[0]
                        old_img_h = img.shape[2]
                        old_img_w = img.shape[3]
                        for i in range(3):
                            model(img, augment=augment)[0]

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=augment)[0]
                    t2 = time_synchronized()

                    # Apply NMS
                    pred = non_max_suppression(pred, self.threshold, self.iou_threshold, classes=None,
                                               agnostic=False)
                    t3 = time_synchronized()

                    # Apply Classifier
                    if classify:
                        pred = apply_classifier(pred, modelc, img, im0s)

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                        p = Path(p)  # to Path
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            dets_to_sort = np.empty((0, 6))
                            # NOTE: We send in detected object class too
                            for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                                dets_to_sort = np.vstack((dets_to_sort,
                                                          np.array([x1, y1, x2, y2, conf, detclass])))

                            tracked_dets = sort_tracker.update(dets_to_sort, False)
                            tracks = sort_tracker.getTrackers()

                            # draw boxes for visualization
                            if len(tracked_dets) > 0:
                                bbox_xyxy = tracked_dets[:, :4]
                                identities = tracked_dets[:, 8]
                                categories = tracked_dets[:, 4]
                                confidences = None

                                if self.show_track:
                                    # loop over tracks
                                    for t, track in enumerate(tracks):
                                        track_color = colors[int(track.detclass)] if not False else \
                                        sort_tracker.color_list[t]

                                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                                        int(track.centroidarr[i][1])),
                                                  (int(track.centroidarr[i + 1][0]),
                                                   int(track.centroidarr[i + 1][1])),
                                                  track_color, thickness=self.thickness)
                                         for i, _ in enumerate(track.centroidarr)
                                         if i < len(track.centroidarr) - 1]


                            im0, text_results = self.draw_boxes(im0, bbox_xyxy, text_results, identities, categories, confidences, names, colors)

                        # Print time (inference + NMS)
                        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                        convertir_QT = QImage(im0.data, im0.shape[1], im0.shape[0],
                                              QImage.Format_RGB888)
                        pic = convertir_QT.scaled(600, 480, Qt.KeepAspectRatioByExpanding)
                        self.Imageupd.emit(pic)
                        self.Labelupd.emit(text_results)

        except Exception as e:
            return None

    def stop(self):
        self.quit()
        self.running = False

    def pause(self):
        self.pause_video = True

    def play(self):
        self.pause_video = False