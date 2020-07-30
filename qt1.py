
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QGridLayout, QSizePolicy, QListWidget, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


class WinForm(QWidget):
	def __init__(self,parent=None):
		super(WinForm,self).__init__(parent)
		self.initUI()
	def initUI(self):
		grid = QGridLayout()
		grid.setSpacing(10)
		fundlist=QListWidget(self)
		fundlist.addItem("item1")
		fundlist.addItem("item2")
		fundlist.addItem("item3")
		fundlist.addItem("item5")
		grid.addWidget(fundlist,1,0)
		chart=chartcanvas()
		grid.addWidget(chart,1,1)
		self.show()

class chartcanvas(FigureCanvas):
	def __init__(self):
		self.fig=Figure()
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.plot()
	def plot(self):
		p1=self.fig.add_subplot(1,1,1)
		p1.plot(np.linspace(1,10,200),np.sin(np.linspace(1,10,200)))
		self.draw()

if __name__ == '__main__':
	app=QApplication([])
	form=WinForm()
	form.show()
	sys.exit(app.exec_())


