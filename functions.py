from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from models.experimental import attempt_load
from utils.general import check_img_size
import torch

def load_model():
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load("model-epp/model.pt", map_location=device)  # load FP32 model

    trace = True
    if trace:
        model = TracedModel(model, device, 640)

    if half:
        model.half()  # to FP16

    return model

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

def epp_state(bbox, categories=None, names=None):
    result_state = ''
    results = ''
    pos_people = []
    pos_hardhat = []
    pos_vest = []
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]

        cat = categories[i]
        id = i
        # conf = confidences[i] if confidences is not None else 0
        if cat == 'person':
            pos_people.append([x1, y1, x2, y2, id])
        if cat == 'hardhat':
            pos_hardhat.append([x1, y1, x2, y2, id])
        if cat == 'vest':
            pos_vest.append([x1, y1, x2, y2, id])

    if len(pos_people) > 0:
        if len(pos_hardhat) == 0:
            print("There's " + str(len(pos_people)) + " person(people) without hardhat(s)")
            results += "Hay" + str(len(pos_people)) + " persona(s) sin casco" + "\n"

        if len(pos_vest) == 0:
            print("There's " + str(len(pos_people)) + " person(people) without vest")
            results += "Hay " + str(len(pos_people)) + " persona(s) sin chaleco" + "\n"

        for i in range(len(pos_people)):
            check_hardhat = 0
            check_vest = 0
            imH = int(pos_people[i][3] - pos_people[i][1])
            imW = int(pos_people[i][2] - pos_people[i][0])

            if len(pos_hardhat) > 0:
                num_hardhat = len(pos_hardhat)
                for j in range(len(pos_hardhat)):
                    hh_xmindiff = abs(pos_people[i][0] - pos_hardhat[j][0])
                    hh_ymindiff = abs(pos_people[i][1] - pos_hardhat[j][1])

                    if ((hh_ymindiff) >= 0 and (hh_ymindiff) < (20 * imH / 100) and (
                            hh_xmindiff) > (1 * imW / 100) and (hh_xmindiff) < (50 * imW / 100)):
                        print("Hardhat " + str(pos_hardhat[j][4]) + " belong to person " + str(pos_people[i][4]) + "\n")
                        results += "Casco " + str(pos_hardhat[j][4]) + " pertenece a la persona " + str(
                            pos_people[i][4]) + "\n"
                        check_hardhat = check_hardhat + 1
                    else:
                        num_hardhat = num_hardhat - 1
                if num_hardhat == 0:
                    print("Person " + str(pos_people[i][4]) + " doesn't have hardhat")
                    results += "Persona " + str(pos_people[i][4]) + " no tiene casco" + "\n"

            if len(pos_vest) > 0:
                num_vest = len(pos_vest)
                for j in range(len(pos_vest)):
                    hh_xmindiff = abs(pos_people[i][0] - pos_vest[j][0])
                    hh_ymindiff = abs(pos_people[i][1] - pos_vest[j][1])
                    # hh_ymaxdiff = abs(box_detected[pos_people[i]][0] - box_detected[pos_vest[j]][2])

                    if ((hh_ymindiff) > (10 * imH / 100) and (hh_ymindiff) < (50 * imH / 100) and (
                            hh_xmindiff) >= 0 and (hh_xmindiff) < (30 * imW / 100)):
                        print("Vest " + str(pos_vest[j][4]) + " belongs to person " + str(pos_people[i][4]))
                        results += "Chaleco " + str(pos_vest[j][4]) + " pertenece a la persona " + str(
                            pos_people[i][4]) + "\n"
                        check_vest = check_vest + 1
                    else:
                        num_vest = num_vest - 1
                if num_vest == 0:
                    print("Person " + str(pos_people[i][4]) + " doesn't have vest")
                    results += "Persona " + str(pos_people[i][4]) + " no tiene chaleco" + "\n"

            if (check_hardhat > 0 and check_vest > 0):
                result_state += "Persona " + str(pos_people[i][4]) + ": Completo" + "\n"
            else:
                result_state += "Persona " + str(pos_people[i][4]) + ": Incompleto" + "\n"
    else:
        print("No people")
        results += "No hay personas"
    return results, result_state

def epp_state_track(bbox, identities=None, categories=None, confidences=None, names=None):
    result_state = ''
    results = ''
    pos_people = []
    pos_hardhat = []
    pos_vest = []
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0
        if names[cat] == 'person':
            pos_people.append([x1,y1,x2,y2,id])
        if names[cat] == 'hardhat':
            pos_hardhat.append([x1,y1,x2,y2,id])
        if names[cat] == 'vest':
            pos_vest.append([x1,y1,x2,y2,id])

    if len(pos_people) > 0:
        if len(pos_hardhat) == 0:
            print("There's " + str(len(pos_people)) + " person(people) without hardhat(s)")
            results += "Hay" + str(len(pos_people)) + " persona(s) sin casco" + "\n"

        if len(pos_vest) == 0:
            print("There's " + str(len(pos_people)) + " person(people) without vest")
            results += "Hay " + str(len(pos_people)) + " persona(s) sin chaleco" + "\n"

        for i in range(len(pos_people)):
            check_hardhat = 0
            check_vest = 0
            imH = int(pos_people[i][3] - pos_people[i][1])
            imW = int(pos_people[i][2] - pos_people[i][0])

            if len(pos_hardhat) > 0:
                num_hardhat = len(pos_hardhat)
                my_hardhat = []
                for j in range(len(pos_hardhat)):
                    hh_xmindiff = abs(pos_people[i][0] - pos_hardhat[j][0])
                    hh_ymindiff = abs(pos_people[i][1] - pos_hardhat[j][1])
                    iou = bb_intersection_over_union(pos_people[i][:-1], pos_hardhat[j][:-1])

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
                        pos_people[i][4]) + "\n"
                elif check_hardhat == 1:
                    results += "Casco " + str(my_hardhat[0][4]) + " pertenece a la persona " + str(
                        pos_people[i][4]) + "\n"

                if num_hardhat == 0:
                    print("Person " + str(pos_people[i][4]) + " doesn't have hardhat")
                    results += "Persona " + str(pos_people[i][4]) + " no tiene casco" + "\n"


            if len(pos_vest) > 0:
                num_vest = len(pos_vest)
                my_vest = []
                for j in range(len(pos_vest)):
                    hh_xmindiff = abs(pos_people[i][0] - pos_vest[j][0])
                    hh_ymindiff = abs(pos_people[i][1] - pos_vest[j][1])
                    iou = bb_intersection_over_union(pos_people[i][:-1], pos_vest[j][:-1])

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
                        pos_people[i][4]) + "\n"
                elif check_vest == 1:
                    results += "Chaleco " + str(my_vest[0][4]) + " pertenece a la persona " + str(
                        pos_people[i][4]) + "\n"

                if num_vest == 0:
                    print("Person " + str(pos_people[i][4]) + " doesn't have vest")
                    results += "Persona " + str(pos_people[i][4]) + " no tiene chaleco" + "\n"

            if (check_hardhat>0 and check_vest>0):
                result_state += "Persona " + str(pos_people[i][4]) + ": Completo" + "\n"
            else:
                result_state += "Persona " + str(pos_people[i][4]) + ": Incompleto" + "\n"
    else:
        print("No people")
        results += "No hay personas"
    return results, result_state