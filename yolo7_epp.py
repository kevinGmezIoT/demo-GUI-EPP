from PyQt5.QtCore import *
from PyQt5.QtGui import *

from sort import *
from threading import Thread

from functions import IaCls

ia_ = IaCls('model-epp')

class Work_image(QThread):
    Imageupd = pyqtSignal(QImage)
    Labelupd = pyqtSignal(str)
    Stateupd = pyqtSignal(str)
    threshold = 0.25
    iou_threshold = 0.45
    thickness = False
    nobbox = False

    def __init__(self, path):
        super().__init__()
        self.im_path = path

    def run(self):
        try:
            ia_.start_detection(type='image', threshold=self.threshold, iou_threshold=self.iou_threshold,
                            Imageupd=self.Imageupd,
                            Labelupd=self.Labelupd, Stateupd=self.Stateupd, thickness=self.thickness, url_base=self.im_path)

        except Exception as e:
            return None



class Work_track(QThread):
    Imageupd = pyqtSignal(QImage)
    Labelupd = pyqtSignal(str)
    Stateupd = pyqtSignal(str)
    threshold = 0.25
    iou_threshold = 0.45
    thickness = False
    nobbox = False
    show_track = False

    def __init__(self, path):
        super().__init__()
        self.video_path = path

    def run(self):
        try:
            sort_tracker = Sort(max_age=5,
                                min_hits=2,
                                iou_threshold=0.2)

            ia_.start_detection(type='video', threshold=self.threshold, iou_threshold=self.iou_threshold,
                            Imageupd=self.Imageupd,
                            Labelupd=self.Labelupd, Stateupd=self.Stateupd, thickness=self.thickness,
                            show_track=self.show_track, url_base=self.video_path,
                            sort_tracker=sort_tracker)

        except Exception as e:
            return None

    def stop(self):
        ia_.stop()
        ia_.__del__()

    def pause(self):
        ia_.pause()

    def play(self):
        ia_.play()

class Work_cam(QThread):
    Imageupd = pyqtSignal(QImage)
    Labelupd = pyqtSignal(str)
    Stateupd = pyqtSignal(str)
    threshold = 0.25
    iou_threshold = 0.45
    thickness = False
    nobbox = False
    show_track = False

    def __init__(self, path, user, pwd):
        super().__init__()
        index = path.find("://") + 3
        credentials = user + ":" + pwd + "@"
        self.url_base = path[:index] + credentials + path[index:]
        self.stop_cam = False

    def run(self):
        try:
            sort_tracker = Sort(max_age=5,
                                min_hits=2,
                                iou_threshold=0.2)

            ia_.start_detection(type='cam', threshold=self.threshold, iou_threshold=self.iou_threshold, Imageupd=self.Imageupd,
                            Labelupd=self.Labelupd, Stateupd=self.Stateupd, thickness=self.thickness, show_track=self.show_track, url_base=self.url_base,
                            sort_tracker=sort_tracker)

        except Exception as e:
            return None

    def stop(self):
        ia_.stop()
        ia_.__del__()

    def pause(self):
        ia_.pause()

    def play(self):
        ia_.play()
