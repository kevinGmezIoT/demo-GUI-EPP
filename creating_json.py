import shortuuid
import uuid
import json

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

dict_api_to_send['userId'] = 'vytxeTZskVKR7C7WgdSP3d'
dict_api_to_send['userKey'] = str(uuid.uuid4())
dict_api_to_send['camera']['cameraId'] = shortuuid.uuid()
dict_api_to_send['camera']['width'] = 1024
dict_api_to_send['camera']['height'] = 720
dict_api_to_send['frame']['frameId'] = shortuuid.uuid()
dict_api_to_send['frame']['sendAt'] = '2021-09-16T03:53:36.125Z'
dict_api_to_send['frame']['filename'] = '1631764416.jpg'
dict_api_to_send['frame']['path'] = '<cameraId>/2021/09/16'
dict_api_to_send['frame']['model']['version'] = 'yolov7-det-aurus-v1.0'
dict_api_to_send['frame']['model']['framework'] = 'yolov7'
dict_api_to_send['frame']['model']['type'] = 'tracking'
dict_api_to_send['frame']['model']['labels'] = ['person', 'hardhat', 'vest', 'glasses', 'gloves', 'with_mask', 'without_mask']
dict_api_to_send['frame']['features']['settings']['maxPerson'] = 5
dict_api_to_send['frame']['features']['settings']['minPerson'] = 1
dict_api_to_send['frame']['features']['settings']['eppNeeded'] = ['hardhat', 'vest']
dict_api_to_send['frame']['features']['settings']['showOneBox'] = False

for i in range(3):
    dict_detections['name'] = 'person'
    dict_detections['amount'] = 2
    dict_detections['objects'] = []
    for j in range(2):
        dict_person_objects['id'] = '10'
        dict_person_objects['score'] = 0.5
        dict_person_objects['boundingBox']['x1'] = 100
        dict_person_objects['boundingBox']['y1'] = 20
        dict_person_objects['boundingBox']['x2'] = 110
        dict_person_objects['boundingBox']['x2'] = 40
        dict_detections['objects'].append(dict_person_objects)
    dict_api_to_send['frame']['features']['detections'].append(dict_detections)
print('')
json_api_to_send = json.dumps(dict_api_to_send)
print(json_api_to_send)
