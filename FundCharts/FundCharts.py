import FundNetv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QHBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QListWidgetItem, QGridLayout, QListWidget, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

class FundChart(QWidget):
	def __init__(self,fundmodel):
		super(FundChart, self).__init__()
		self.threadPool=ThreadPoolExecutor(max_workers=5, thread_name_prefix="thread_")
		self.fund=fundmodel
		self.initUI()
	def initUI(self):
		self.gridlayout=QGridLayout()
		self.gridlayout.setSpacing(5)
		self.fundlist=QListWidget(self)
		self.fundlist.setMinimumWidth(240)
		self.fundlist.setMaximumWidth(240)
		items=self.fund.getitems()
		__it=iter(items)
		for item in __it :
			__item=QListWidgetItem("{} {}".format(item,items[item]))
			__item.data=item
			self.fundlist.addItem(__item)
		self.fundlist.itemClicked.connect(self.selectItem)
		self.gridlayout.addWidget(self.fundlist,1,0)
		self.refreshbtn=QPushButton("Refresh")
		self.refreshbtn.setMaximumWidth(100)
		self.refreshbtn.clicked.connect(self.refreshbtnclick)
		self.gridlayout.addWidget(self.refreshbtn,2,0)
		self.figure=plt.figure()
		self.canvas=FigureCanvas(self.figure)
		self.gridlayout.addWidget(self.canvas,1,1,2,1)
		self.canvas.draw()
		self.setLayout(self.gridlayout)
	def selectItem(self,obj):
		self.figure.clf()
		self.fund.showfigure(obj.data,figure=self.figure)
		self.canvas.draw()
	def refreshbtnclick(self):
		future=self.threadPool.submit(self.fund.refreshcdfma,45)
		future.add_done_callback(self.callback_refreshbtnclick)
	def callback_refreshbtnclick(self,future):
		self.fundlist.clear()
		items=self.fund.getitems()
		__it=iter(items)
		for item in __it :
			__item=QListWidgetItem("{} {}".format(item,items[item]))
			__item.data=item
			self.fundlist.addItem(__item)
class manipulate_panel(QWidget):
	def __init__(self,fundmodel):
		super(manipulate_panel, self).__init__()
		self.fund=fundmodel
		self.initUI()
	def initUI(self):
		self.hboxlayout=QHBoxLayout()
		self.hboxlayout.setSpacing(6)
		self.fundcodeedit=QLineEdit()
		self.fundcodeedit.setMaximumWidth(100)
		self.fundcodeedit.setMaxLength(6)
		self.fundcodeedit.setInputMask('999999')
		self.grabbtn=QPushButton("PullData")
		self.grabbtn.setMaximumWidth(100)
		self.grabbtn.clicked.connect(self.grabbtnclick)
		self.grabnetvbtn=QPushButton("AddFund")
		self.grabnetvbtn.setMaximumWidth(100)
		self.grabnetvbtn.clicked.connect(self.grabnetvbtnclick)
		self.hboxlayout.addWidget(self.fundcodeedit,alignment=Qt.AlignLeft)
		self.hboxlayout.addWidget(self.grabnetvbtn,alignment=Qt.AlignLeft)
		self.hboxlayout.addWidget(self.grabbtn,alignment=Qt.AlignLeft)
		self.setLayout(self.hboxlayout)
	def grabbtnclick(self):
		self.fund.grabfromdb()
	def grabnetvbtnclick(self):
		fundcode="{}".format(self.fundcodeedit.text())
		if len(fundcode) == 6 :
			print("FundCode: {}".format(fundcode))
			self.fund.grabnetv(fundcode)
		self.fundcodeedit.clear()
		self.fundcodeedit.setFocus()
class MainUI(QWidget):
	def __init__(self):
		super(MainUI, self).__init__()
		self.fund=FundNetv.fundnetv()
		self.initUI()
	def initUI(self):
		self.vboxlayout=QVBoxLayout()
		self.vboxlayout.setSpacing(5)
		self.fundchart=FundChart(self.fund)
		self.vboxlayout.addWidget(self.fundchart)
		self.manipulatepanel=manipulate_panel(self.fund)
		self.vboxlayout.addWidget(self.manipulatepanel)
		self.setLayout(self.vboxlayout)

if __name__ == '__main__':
	app=QApplication([])
	ui=MainUI()
	ui.show()
	sys.exit(app.exec_())
