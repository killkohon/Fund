import FundNetv
from flask import Flask, render_template, make_response
from flask_bootstrap import Bootstrap
import io
import numpy as np
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from wsgiref.simple_server import make_server

class FCApp(Flask):
    def __init__(self,name):
        super().__init__(name,static_url_path='')
        self.Model = FundNetv.fundnetv()
        self.bootstrap = Bootstrap(self)

if __name__ == '__main__':
    flaskapp=FCApp(__name__)
    @flaskapp.route('/test/<string:fundcode>')
    def chartsTest(fundcode):
        return render_template('plot.html', name='test', url='/plot'+fundcode+'.png')

    @flaskapp.route('/plot<string:fundcode>.png')
    def charttest(fundcode):
        fig=plt.figure()
        subplot=fig.add_subplot(1,1,1)
        x=np.linspace(1,10,100)
        y=np.sin(x)
        subplot.plot(x,y)
        flaskapp.Model.showfigure(tickcode=fundcode,figure=fig)
        output=figure2canvas(fig)
        response = make_response(output.getvalue())
        response.mimetype = 'image/png'
        return response

    def figure2canvas(fig):
        canvas = FigureCanvas(fig)
        output = io.BytesIO()
        canvas.print_png(output)
        return output
    
    srv = make_server('0.0.0.0', 8080, flaskapp)
    srv.serve_forever()
