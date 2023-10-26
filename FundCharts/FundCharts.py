import FundNetv
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import datetime as dt


class FundChart(QWidget):
    def __init__(self, fundmodel):
        super(FundChart, self).__init__()
        self.threadPool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="Qthread_")
        self.fund = fundmodel
        self.initUI()

    def initUI(self):
        self.gridlayout = QGridLayout()
        self.gridlayout.setSpacing(5)
        self.fundlist = QListWidget(self)
        self.fundlist.setMinimumWidth(240)
        self.fundlist.setMaximumWidth(240)
        self.highlightbrush = QBrush(Qt.SolidPattern)
        self.highlightbrush.setColor(QColor("cyan"))
        items = self.fund.getitems(3)
        __it = iter(items)
        for item in __it:
            __item = QListWidgetItem("{} {}".format(item, items[item][0]))
            __item.data = item
            if items[item][1] == 1:
                __item.setBackground(self.highlightbrush)
            self.fundlist.addItem(__item)
        self.fundlist.itemClicked.connect(self.selectItem)
        self.gridlayout.addWidget(self.fundlist, 1, 0)

        self.btnWidget = QWidget()
        btnGridlayout = QGridLayout()
        btnGridlayout.setSpacing(5)

        self.refreshbtn = QPushButton("Refresh")
        self.refreshbtn.setMaximumWidth(60)
        self.refreshbtn.clicked.connect(self.refreshbtnclick)

        self.diffbtn = QPushButton("ByDiff")
        self.diffbtn.setMaximumWidth(60)
        self.diffbtn.clicked.connect(self.diffbtnclick)

        self.sortByNamebtn = QPushButton("ByName")
        self.sortByNamebtn.setMaximumWidth(60)
        self.sortByNamebtn.clicked.connect(self.sortByNamebtnclick)

        self.sortByCodebtn = QPushButton("ByCode")
        self.sortByCodebtn.setMaximumWidth(60)
        self.sortByCodebtn.clicked.connect(self.sortByCodebtnclick)

        btnGridlayout.addWidget(self.refreshbtn, 1, 0)
        btnGridlayout.addWidget(self.sortByNamebtn, 1, 1)
        btnGridlayout.addWidget(self.sortByCodebtn, 1, 2)
        btnGridlayout.addWidget(self.diffbtn, 2, 0)
        self.btnWidget.setLayout(btnGridlayout)
        self.gridlayout.addWidget(self.btnWidget, 2, 0)

        self.tabs=QTabWidget()

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.tabs.addTab(self.canvas,"Charts")
        self.svdfigure = plt.figure()
        self.svdcanvs=FigureCanvas(self.svdfigure)
        self.tabs.addTab(self.svdcanvs,"SVD")
        self.predictfigure = plt.figure()
        self.predictcanvas=FigureCanvas(self.predictfigure)
        self.tabs.addTab(self.predictcanvas,"Predict")
        self.gridlayout.addWidget(self.tabs, 1, 1)
        self.canvas.draw()
        self.setLayout(self.gridlayout)

    def selectItem(self, obj):
        self.figure.clf()
        self.fund.currentfund = obj.data
        self.fund.showfigure(obj.data, figure=self.figure)
        self.canvas.draw()
        self.svdfigure.clf()
        self.fund.showSVDFigure(obj.data,figure=self.svdfigure)
        self.svdcanvs.draw()
        self.predictfigure.clf()
        self.fund.showPredictFigure(obj.data,figure=self.predictfigure)
        self.predictcanvas.draw()


    def refreshbtnclick(self):
        future = self.threadPool.submit(self.fund.refreshcdfma, 45)
        future.add_done_callback(self.callback_refreshbtnclick)

    def diffbtnclick(self):
        print("Diff Clicked")
        future = self.threadPool.submit(self.fund.updatestddiff, 45)
        future.add_done_callback(self.callback_updatebtnclick)

    def callback_refreshbtnclick(self, future):
        self.fundlist.clear()
        items = self.fund.getitems(1)
        __it = iter(items)
        for item in __it:
            __item = QListWidgetItem("{} {}".format(item, items[item][0]))
            __item.data = item
            if items[item][1] == 1:
                __item.setBackground(self.highlightbrush)
            self.fundlist.addItem(__item)
        print("Refreshed at {}".format(dt.datetime.now()))

    def callback_updatebtnclick(self, future):
        self.fundlist.clear()
        items = self.fund.getitems(4)
        __it = iter(items)
        for item in __it:
            __item = QListWidgetItem("{} {}".format(item, items[item][0]))
            __item.data = item
            if items[item][1] == 1:
                __item.setBackground(self.highlightbrush)
            self.fundlist.addItem(__item)
        print("update std diff at {}".format(dt.datetime.now()))

    def sortByNamebtnclick(self):
        self.fundlist.clear()
        items = self.fund.getitems(2)
        __it = iter(items)
        for item in __it:
            __item = QListWidgetItem("{} {}".format(item, items[item][0]))
            __item.data = item
            if items[item][1] == 1:
                __item.setBackground(self.highlightbrush)
            self.fundlist.addItem(__item)

    def sortByCodebtnclick(self):
        self.fundlist.clear()
        items = self.fund.getitems(3)
        __it = iter(items)
        for item in __it:
            __item = QListWidgetItem("{} {}".format(item, items[item][0]))
            __item.data = item
            if items[item][1] == 1:
                __item.setBackground(self.highlightbrush)
            self.fundlist.addItem(__item)


class manipulate_panel(QWidget):
    def __init__(self, fundmodel):
        super(manipulate_panel, self).__init__()
        self.fund = fundmodel
        self.initUI()

    def initUI(self):
        self.hboxlayout = QHBoxLayout()
        self.hboxlayout.setSpacing(6)
        self.fundcodeedit = QLineEdit()
        self.fundcodeedit.setMaximumWidth(100)
        self.fundcodeedit.setMaxLength(6)
        self.fundcodeedit.setInputMask('999999')
        self.grabbtn = QPushButton("PullData")
        self.grabbtn.setMaximumWidth(100)
        self.grabbtn.clicked.connect(self.grabbtnclick)
        self.grabnetvbtn = QPushButton("AddFund")
        self.grabnetvbtn.setMaximumWidth(100)
        self.holdingChkbox = QCheckBox("holding", self)
        self.holdingChkbox.stateChanged.connect(self.holdingchkboxClick)
        self.grabnetvbtn.clicked.connect(self.grabnetvbtnclick)
        self.hboxlayout.addWidget(self.fundcodeedit, alignment=Qt.AlignLeft)
        self.hboxlayout.addWidget(self.grabnetvbtn, alignment=Qt.AlignLeft)
        self.hboxlayout.addWidget(self.grabbtn, alignment=Qt.AlignLeft)
        self.hboxlayout.addWidget(self.holdingChkbox, alignment=Qt.AlignLeft)
        self.setLayout(self.hboxlayout)

    def grabbtnclick(self):
        self.fund.grabfromdb()

    def grabnetvbtnclick(self):
        fundcode = "{}".format(self.fundcodeedit.text())
        if len(fundcode) == 6:
            print("FundCode: {}".format(fundcode))
            self.fund.grabnetv(fundcode)
        self.fundcodeedit.clear()
        self.fundcodeedit.setFocus()

    def holdingchkboxClick(self, state):
        if state == Qt.Checked:
            print("checked-{}".format(self.fund.currentfund))
            self.fund.setholding(self.fund.currentfund, True)
        else:
            print("unchecked-{}".format(self.fund.currentfund))
            self.fund.setholding(self.fund.currentfund, False)


class MainUI(QWidget):
    def __init__(self):
        super(MainUI, self).__init__()
        self.fund = FundNetv.fundnetv()
        self.initUI()

    def initUI(self):
        self.vboxlayout = QVBoxLayout()
        self.vboxlayout.setSpacing(5)
        
        self.fundchart = FundChart(self.fund)
        self.vboxlayout.addWidget(self.fundchart)
        self.manipulatepanel = manipulate_panel(self.fund)
        self.vboxlayout.addWidget(self.manipulatepanel)
        self.setLayout(self.vboxlayout)


if __name__ == '__main__':
    app = QApplication([])
    ui = MainUI()
    ui.show()
    sys.exit(app.exec_())
