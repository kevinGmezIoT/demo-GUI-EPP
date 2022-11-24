# Importing needed libraries
# We need sys library to pass arguments into QApplication
import sys
# QtWidgets to work with widgets
from PyQt5 import QtWidgets
# QPixmap to work with images
from PyQt5.QtGui import QPixmap

# Importing designed GUI in Qt Designer as module
import d_menu as design

from g_retail import RetailApp
from g_epp import EppApp

# Creating main class to connect objects in designed GUI with useful code
# Passing as arguments widgets of main window
# and main class of created design that includes all created objects in GUI
class MainApp(QtWidgets.QMainWindow, design.Ui_MenuWindow):
    # Constructor of the class
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Connecting event of clicking on the button with needed function
        self.b_retail.clicked.connect(self.openRetailWindow)
        self.b_epp.clicked.connect(self.openEppWindow)

    def openRetailWindow(self):
        self.ui = RetailApp()
        self.ui.showMaximized()

    def openEppWindow(self):
        self.ui = EppApp()
        self.ui.showMaximized()


# Defining main function to be run
def main():
    # Initializing instance of Qt Application
    app = QtWidgets.QApplication(sys.argv)

    # Initializing object of designed GUI
    window = MainApp()

    # Showing designed GUI
    window.show()

    # Running application
    app.exec_()


"""
End of: 
Main function
"""


# Checking if current namespace is main, that is file is not imported
if __name__ == '__main__':
    # Implementing main() function
    main()
