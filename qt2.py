from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QListWidgetItem
import sys
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QPushButton, QGridLayout, QSizePolicy, QListWidget, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class FundChart(QWidget):
	def __init__(self):
		super(FundChart, self).__init__()
		self.initUI()
	def initUI(self):
		self.gridlayout=QGridLayout()
		self.gridlayout.setSpacing(10)
		self.fundlist=QListWidget(self)
		self.fundlist.setMinimumWidth(220)
		self.fundlist.setMaximumWidth(220)
		item1=QListWidgetItem("item1")
		item1.data=1
		self.fundlist.addItem(item1)
		item2=QListWidgetItem("item2")
		item2.data=2
		self.fundlist.addItem(item2)
		self.fundlist.itemClicked.connect(self.selectItem)
		self.gridlayout.addWidget(self.fundlist,1,0)
		self.figure=plt.figure()
		'''
		self.p1=self.figure.add_subplot(1,1,1)
		self.p1.plot(np.linspace(0,12,300),np.sin(np.linspace(0,12,300)))
		self.p1.grid()
		'''
		self.canvas=FigureCanvas(self.figure)
		self.gridlayout.addWidget(self.canvas,1,1)
		self.canvas.draw()
		self.setLayout(self.gridlayout)
	def selectItem(self,obj):
		if obj.data == 2 :
			self.figure.clf()
			self.p1=self.figure.add_subplot(1,1,1)
			self.p1.plot(np.linspace(0,12,300),np.sin(np.linspace(0,12,300)))
			self.p1.grid()
			self.canvas.draw()
		else:
			self.figure.clf()
			self.p1=self.figure.add_subplot(1,1,1)
			self.p1.plot(np.linspace(0,12,300),np.cos(np.linspace(0,12,300)))
			self.p1.grid()
			self.canvas.draw()

if __name__ == '__main__':
	app=QApplication([])
	ui=FundChart()
	ui.show()
	sys.exit(app.exec_())