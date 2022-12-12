from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.plots import plot_one_box
from numpy import random
from pathlib import Path
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import cv2
import time
import shortuuid
import uuid
import json

def creating_json(userId, userKey, cameraID, frameId, filename, path_to_file, frame, model_info, feature_settings, detection_results):
    dict_person_objects = {
        'id': '',
        'score': 0,
        'boundingBox':
            {
                'x1': 0,
                'y1': 0,
                'x2': 0,
                'y2': 0
            },
        'associatedFeaturesId': [],
        'isEppComplete': False
    }

    dict_other_objects = {
        'id': '',
        'score': 0,
        'boundingBox':
            {
                'x1': 0,
                'y1': 0,
                'x2': 0,
                'y2': 0
            },
    }

    dict_detections = {
        'name': '',
        'amount': 0,
        'objects': []
    }

    dict_api_to_send = {
        'userId': '',
        'userKey': '',
        'camera':
            {
                'cameraId': '',
                'width': 0,
                'height': 0
            },
        'frame':
            {
                'frameId': '',
                'sendAt': '',
                'filename': '',
                'path': '',
                'model':
                    {
                        'version': '',
                        'framework': '',
                        'type': '',
                        'labels': []
                    },
                'features':
                    {
                        'settings':
                            {
                                'maxPerson': 0,
                                'minPerson': 0,
                                'eppNeeded': [],
                                'showOneBox': False
                            },
                        'detections': []
                    }
            }
    }

    dict_api_to_send['userId'] = userId
    dict_api_to_send['userKey'] = userKey
    dict_api_to_send['camera']['cameraId'] = cameraID
    dict_api_to_send['camera']['width'] = frame.shape[1]
    dict_api_to_send['camera']['height'] = frame.shape[0]
    dict_api_to_send['frame']['frameId'] = frameId
    dict_api_to_send['frame']['sendAt'] = '2021-09-16T03:53:36.125Z'
    dict_api_to_send['frame']['filename'] = filename
    dict_api_to_send['frame']['path'] = path_to_file
    dict_api_to_send['frame']['model']['version'] = model_info.version
    dict_api_to_send['frame']['model']['framework'] = model_info.framework
    dict_api_to_send['frame']['model']['type'] = model_info.type
    dict_api_to_send['frame']['model']['labels'] = model_info.labels
    dict_api_to_send['frame']['features']['settings']['maxPerson'] = feature_settings.maxPerson
    dict_api_to_send['frame']['features']['settings']['minPerson'] = feature_settings.minPerson
    dict_api_to_send['frame']['features']['settings']['eppNeeded'] = feature_settings.eppNeeded
    dict_api_to_send['frame']['features']['settings']['showOneBox'] = feature_settings.showOneBox

    for i in range(3):
        dict_detections['name'] = detection_results.name
        dict_detections['amount'] = detection_results.amount
        dict_detections['objects'] = []
        for j in range(2):
            dict_person_objects['id'] = detection_results.id
            dict_person_objects['score'] = detection_results.score
            dict_person_objects['boundingBox']['x1'] = detection_results.x1
            dict_person_objects['boundingBox']['y1'] = detection_results.y1
            dict_person_objects['boundingBox']['x2'] = detection_results.x2
            dict_person_objects['boundingBox']['y2'] = detection_results.y2
            dict_detections['objects'].append(dict_person_objects)
        dict_api_to_send['frame']['features']['detections'].append(dict_detections)


def save_frame():
    return

def convert_image(im0, Imageupd, Stateupd, Labelupd, results, result_state):
    convertir_QT = QImage(im0.data, im0.shape[1], im0.shape[0], im0.shape[2] * im0.shape[1],
                          QImage.Format_RGB888)
    pic = convertir_QT.scaled(640, 480, Qt.KeepAspectRatioByExpanding)
    Imageupd.emit(pic)
    Labelupd.emit(results)
    Stateupd.emit(result_state)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def analytics(bbox, identities=None, categories=None, names=None):
    result_state = ''
    results = ''
    pos_people = []
    pos_hardhat = []
    pos_vest = []

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]

        if identities is not None:
            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else i
        else:
            cat = categories[i] if categories is not None else ''
            id = i

        # conf = confidences[i] if confidences is not None else 0
        if names is not None:
            if names[cat] == 'person':
                pos_people.append([x1, y1, x2, y2, id])
            if names[cat] == 'hardhat':
                pos_hardhat.append([x1, y1, x2, y2, id])
            if names[cat] == 'vest':
                pos_vest.append([x1, y1, x2, y2, id])
        else:
            if cat == 'person':
                pos_people.append([x1, y1, x2, y2, id])
            if cat == 'hardhat':
                pos_hardhat.append([x1, y1, x2, y2, id])
            if cat == 'vest':
                pos_vest.append([x1, y1, x2, y2, id])

    if len(pos_people) > 0:
        if len(pos_hardhat) == 0:
            results += "Hay" + str(len(pos_people)) + " persona(s) sin casco" + "\n"

        if len(pos_vest) == 0:
            results += "Hay " + str(len(pos_people)) + " persona(s) sin chaleco" + "\n"

        for p in range(len(pos_people)):
            check_hardhat = 0
            check_vest = 0
            imH = int(pos_people[p][3] - pos_people[p][1])
            imW = int(pos_people[p][2] - pos_people[p][0])

            if len(pos_hardhat) > 0:
                num_hardhat = len(pos_hardhat)
                my_hardhat = []
                for j in range(len(pos_hardhat)):
                    hh_xmindiff = abs(pos_people[p][0] - pos_hardhat[j][0])
                    hh_ymindiff = abs(pos_people[p][1] - pos_hardhat[j][1])
                    iou = bb_intersection_over_union(pos_people[p][:-1], pos_hardhat[j][:-1])

                    if ((hh_ymindiff) >= 0 and (hh_ymindiff) < (40 * imH / 100) and (
                            hh_xmindiff) > (1 * imW / 100) and (hh_xmindiff) < (80 * imW / 100) and iou > 0):
                        current_hardhat = pos_hardhat[j]
                        current_hardhat.append(iou)
                        my_hardhat.append(current_hardhat)
                        check_hardhat = check_hardhat + 1
                    else:
                        num_hardhat = num_hardhat - 1

                if check_hardhat > 1:
                    max_iou = 0
                    id = 0
                    for hardhat in my_hardhat:
                        if hardhat[5] > max_iou:
                            max_iou = hardhat[5]
                            id = hardhat[4]
                    results += "Casco " + str(id) + " pertenece a la persona " + str(
                        pos_people[p][4]) + "\n"
                elif check_hardhat == 1:
                    results += "Casco " + str(my_hardhat[0][4]) + " pertenece a la persona " + str(
                        pos_people[p][4]) + "\n"

                if num_hardhat == 0:
                    results += "Persona " + str(pos_people[p][4]) + " no tiene casco" + "\n"

            if len(pos_vest) > 0:
                num_vest = len(pos_vest)
                my_vest = []
                for j in range(len(pos_vest)):
                    hh_xmindiff = abs(pos_people[p][0] - pos_vest[j][0])
                    hh_ymindiff = abs(pos_people[p][1] - pos_vest[j][1])
                    iou = bb_intersection_over_union(pos_people[p][:-1], pos_vest[j][:-1])

                    if ((hh_ymindiff) >= (0 * imH / 100) and (hh_ymindiff) < (50 * imH / 100) and (
                            hh_xmindiff) >= 0 and (hh_xmindiff) < (50 * imW / 100) and iou > 0):
                        current_vest = pos_vest[j]
                        current_vest.append(iou)
                        my_vest.append(current_vest)
                        check_vest = check_vest + 1
                    else:
                        num_vest = num_vest - 1

                if check_vest > 1:
                    max_iou = 0
                    id = 0
                    for vest in my_vest:
                        if vest[5] > max_iou:
                            max_iou = vest[5]
                            id = vest[4]
                    results += "Chaleco " + str(id) + " pertenece a la persona " + str(
                        pos_people[p][4]) + "\n"
                elif check_vest == 1:
                    results += "Chaleco " + str(my_vest[0][4]) + " pertenece a la persona " + str(
                        pos_people[p][4]) + "\n"

                if num_vest == 0:
                    results += "Persona " + str(pos_people[p][4]) + " no tiene chaleco" + "\n"

            if (check_hardhat > 0 and check_vest > 0):
                result_state += "Persona " + str(pos_people[p][4]) + ": Completo" + "\n"
            else:
                result_state += "Persona " + str(pos_people[p][4]) + ": Incompleto" + "\n"
    else:
        results += "No hay personas"
    return results, result_state

def translate_name(names, cls):
    if names[int(cls)] == 'person':
        name = 'persona'
    if names[int(cls)] == 'hardhat':
        name = 'casco'
    if names[int(cls)] == 'vest':
        name = 'chaleco'
    if names[int(cls)] == 'glasses':
        name = 'lentes'
    if names[int(cls)] == 'gloves':
        name = 'guantes'
    if names[int(cls)] == 'with_mask':
        name = 'mascara'
    if names[int(cls)] == 'without_mask':
        name = 'sin mascara'
    return name

def draw_boxes(img, bbox, thickness= 1, nobbox=False, identities=None, categories=None, confidences=None, names=None,
               colors=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]

        if not nobbox:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)
        name = translate_name(names, cat)
        label = str(id) + ":" + name if identities is not None else f'{name} {confidences[i]:.2f}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img


class IaCls:
    def __init__(self, model_path):
        super().__init__()
        self.model = None
        self.model_path = model_path
        self.stop_ = False
        self.pause_ = False
        self.labels = []

    def load_model(self, device, half):
        # Load model
        model = attempt_load(self.model_path + '/model.pt', map_location=device)  # load FP32 model

        with open(self.model_path + '/classes.names') as f:
            lines = f.readlines()

        for i in range(0, len(lines)):
            self.labels.append(lines[i].rstrip('\n'))

        trace = True
        if trace:
            model = TracedModel(model, device, 640)

        if half:
            model.half()  # to FP16

        return model

    def start_detection(self, type='image', threshold=0.5, iou_threshold=0.2, Imageupd=None, Labelupd=None, Stateupd=None,
                    thickness=False, nobbox=False, show_track=False, url_base=None, sort_tracker=None):
        webcam = url_base.isnumeric() or url_base.endswith('.txt') or url_base.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        with torch.no_grad():
            # Initialize
            set_logging()
            device = select_device('0')
            half = device.type != 'cpu'  # half precision only supported on CUDA

            if self.model == None:
                # Load model
                self.model = self.load_model(device, half)

            stride = int(self.model.stride.max())  # model stride
            imgsz = check_img_size(640, s=stride)  # check img_size

            if type == 'cam':
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(url_base, img_size=imgsz, stride=stride)
            elif type == 'video' or type == 'image':
                dataset = LoadImages(url_base, img_size=imgsz, stride=stride)
            else:
                print('Bad type!')
                return

            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                self.model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.model.parameters())))  # run once
            old_img_w = old_img_h = imgsz
            old_img_b = 1

            for path, img, im0s, vid_cap in dataset:
                if not self.stop_:
                    augment = False
                    if not self.pause_:
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
                                self.model(img, augment=augment)[0]

                        # Inference
                        t1 = time_synchronized()
                        pred = self.model(img, augment=augment)[0]
                        t2 = time_synchronized()

                        # Apply NMS
                        pred = non_max_suppression(pred, threshold, iou_threshold, classes=None,
                                                   agnostic=False)
                        t3 = time_synchronized()

                        # Process detections
                        for i, det in enumerate(pred):  # detections per image
                            if webcam:
                                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                            else:
                                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                            results = ''
                            result_state = ''
                            p = Path(p)  # to Path
                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                                # Print results
                                for c in det[:, -1].unique():
                                    n = (det[:, -1] == c).sum()  # detections per class
                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                                if type != 'image':
                                    score = []
                                    dets_to_sort = np.empty((0, 6))
                                    # NOTE: We send in detected object class too
                                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                                        score.append(conf)
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

                                        if show_track:
                                            # loop over tracks
                                            for t, track in enumerate(tracks):
                                                track_color = colors[int(track.detclass)] if not False else \
                                                    sort_tracker.color_list[t]

                                                [cv2.line(im0, (int(track.centroidarr[i][0]),
                                                                int(track.centroidarr[i][1])),
                                                          (int(track.centroidarr[i + 1][0]),
                                                           int(track.centroidarr[i + 1][1])),
                                                          track_color, thickness=thickness)
                                                 for i, _ in enumerate(track.centroidarr)
                                                 if i < len(track.centroidarr) - 1]

                                    results, result_state = analytics(bbox_xyxy, identities=identities,
                                                                      categories=categories, names=names)
                                    im0 = draw_boxes(im0, bbox_xyxy, thickness, nobbox, identities, categories,
                                                     confidences, names, colors)
                                else:
                                    # Write results
                                    index = 0
                                    box = []
                                    cat = []
                                    score = []
                                    for *xyxy, conf, cls in reversed(det):
                                        im_class = f'{names[int(cls)]}'
                                        name = translate_name(names, cls)
                                        label = f'{name}: {index}'
                                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                                        box.append(xyxy)
                                        cat.append(im_class)
                                        score.append(conf)
                                        index += 1

                                    results, result_state = analytics(bbox=box, categories=cat)
                            # Print time (inference + NMS)
                            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                            convert_image(im0, Imageupd, Stateupd, Labelupd, results, result_state)
                    else:
                        while self.pause_:
                            time.sleep(0.01)
                else:
                    break

    def stop(self):
        self.stop_ = True

    def pause(self):
        self.pause_ = True

    def play(self):
        self.pause_ = False

    def __del__(self):
        self.stop_ = False
        self.pause_ = False
        self.model = None