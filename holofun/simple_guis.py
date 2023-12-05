from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename, askopenfilenames
# from PyQt5.QtWidgets import (QFileDialog, QAbstractItemView, QListView,
#                              QTreeView, QApplication, QDialog)

def openfoldergui(rootdir=None, title=None):
    return _get_opengui(askdirectory, initialdir=rootdir, title=title)

def openfilegui(rootdir=None, title=None):
    return _get_opengui(askopenfilename, initialdir=rootdir, title=title)

def openfilesgui(rootdir=None, title=None):
    return _get_opengui(askopenfilenames,initialdir=rootdir, title=title)

def _get_opengui(openfun, *args, **kwargs):
    root = Tk()
    root.lift()
    root.attributes("-topmost", True)
    root.withdraw()
    dirname = openfun(parent=root, *args, **kwargs)
    root.destroy()
    return dirname

# class getExistingDirectories(QFileDialog):
#     def __init__(self, caption, directory):
#         super(getExistingDirectories, self).__init__(caption=caption, directory=directory)
#         # self.setWindowTitle('Select folders...')
#         self.setFixedSize(1000, 600)
#         self.setOption(self.DontUseNativeDialog, True)
#         self.setFileMode(self.Directory)
#         self.setOption(self.ShowDirsOnly, True)
#         self.findChildren(QListView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)
#         self.findChildren(QTreeView)[0].setSelectionMode(QAbstractItemView.ExtendedSelection)
    
# def openfoldersgui(rootdir=None, title='Select folders...'):
#     qapp = QApplication([])
#     dlg = getExistingDirectories(caption=title, directory=rootdir)
#     if dlg.exec_() == QDialog.Accepted:
#         return dlg.selectedFiles()