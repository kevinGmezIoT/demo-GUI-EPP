import sys
# QtWidgets to work with widgets
from PyQt5 import QtWidgets
# QPixmap to work with images
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import *
from PyQt5.QtGui import *
# Importing designed GUI in Qt Designer as module
import d_epp as design_epp

# Importing YOLO v3 module to Detect Objects on image
from yolo7_epp import run_inference_for_single_image
from yolo7_epp import Work_track, Work_cam
from threading import Thread

"""
Start of:
Main class to add functionality of designed GUI
"""


# Creating main class to connect objects in designed GUI with useful code
# Passing as arguments widgets of main window
# and main class of created design that includes all created objects in GUI
class EppApp(QtWidgets.QMainWindow, design_epp.Ui_MainWindow):
    # Constructor of the class
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        pixmap_image = QPixmap('resources/logo.jpeg')

        # Passing opened image to the Label object
        self.l_logo.setPixmap(pixmap_image)

        # Getting opened image width and height
        # And resizing Label object according to these values
        self.l_logo.resize(pixmap_image.width(), pixmap_image.height())


        # Connecting event of clicking on the button with needed function
        self.b_image.clicked.connect(self.f_image_inference)
        self.b_video.clicked.connect(self.f_video_inference)
        self.b_camera.clicked.connect(self.f_camera_inference)
        self.b_play.clicked.connect(self.playEvent)
        self.b_pause.clicked.connect(self.pauseEvent)

    # Defining function that will be implemented after button is pushed
    # noinspection PyArgumentList
    def f_image_inference(self):
        self.clear()
        # Showing text while image is loading and processing
        self.l_image.setText('Procesando ...')

        # Opening dialog window to choose an image file
        # Giving name to the dialog window --> 'Choose Image to Open'
        # Specifying starting directory --> '.'
        # Showing only needed files to choose from --> '*.png *.jpg *.bmp'
        # noinspection PyCallByClass
        image_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Image to Open',
                                                  '.',
                                                  '*.png *.jpg *.bmp')

        # Slicing only needed full path
        image_path = image_path[0]  # /home/my_name/Downloads/example.png

        # Passing full path to loaded image into YOLO v3 algorithm
        results, text_result, state_result = run_inference_for_single_image(image_path)

        # Opening resulted image with QPixmap class that is used to
        # show image inside Label object
        convertir_QT = QImage(results.data, results.shape[1], results.shape[0], results.shape[2]*results.shape[1],
                              QImage.Format_RGB888)
        pic = convertir_QT.scaled(640, 480, Qt.KeepAspectRatioByExpanding)
        pixmap_image = QPixmap.fromImage(pic)

        # Passing opened image to the Label object
        self.l_image.setPixmap(pixmap_image)

        # Getting opened image width and height
        # And resizing Label object according to these values
        self.l_image.resize(pixmap_image.width(), pixmap_image.height())
        self.l_results.setText(text_result)
        self.l_epp_state.setText(state_result)

    def f_video_inference(self):
        self.clear()
        # Showing text while image is loading and processing
        self.l_image.setText('Procesando ...')

        # Opening dialog window to choose an image file
        # Giving name to the dialog window --> 'Choose Image to Open'
        # Specifying starting directory --> '.'
        # Showing only needed files to choose from --> '*.png *.jpg *.bmp'
        # noinspection PyCallByClass
        video_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Video to Open',
                                                  '.',
                                                  '*.avi *.mp4 *.mpeg4 *.m4v')

        # Slicing only needed full path
        video_path = video_path[0]  # /home/my_name/Downloads/example.png

        self.start_track_video(video_path)

    def start_track_video(self, video_path):
        self.Work = Work_track(video_path)
        self.Work.start()
        self.Work.Imageupd.connect(self.Imageupd_slot)
        self.Work.Labelupd.connect(self.Labelupd_slot)
        self.Work.Stateupd.connect(self.Stateupd_slot)

    def Imageupd_slot(self, Image):
        self.l_image.setPixmap(QPixmap.fromImage(Image))

    def Labelupd_slot(self, output):
        self.l_results.setText(output)

    def Stateupd_slot(self, output):
        self.l_epp_state.setText(output)

    def f_camera_inference(self):
        self.clear()
        self.l_image.setText('Procesando ...')
        self.Work = Work_cam('rtsp://192.168.2.240:7070', 'rubikev', 'PaSsWoRd2874483!')
        self.Work.start()
        self.Work.Imageupd.connect(self.Imageupd_slot)
        self.Work.Labelupd.connect(self.Labelupd_slot)
        self.Work.Stateupd.connect(self.Stateupd_slot)

    def closeEvent(self, event):
        try:
            self.Work.stop()
        except:
            pass

    def playEvent(self):
        self.Work.play()

    def pauseEvent(self):
        self.Work.pause()

    def clear(self):
        try:
            self.Work.stop()
        except:
            pass
        self.l_image.clear()
        self.l_results.clear()
        self.l_epp_state.clear()
