from flask import Flask, render_template, make_response
from flask_bootstrap import Bootstrap
import io
import numpy as np
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas

flaskapp=Flask(__name__)
bootstrap = Bootstrap(flaskapp)

@flaskapp.route('/test')
def chartsTest():
    return render_template('plot.html', name='test', url='/plot.png')

@flaskapp.route('/plot.png')
def charttest():
    fig=plt.figure()
    subplot=fig.add_subplot(1,1,1)
    x=np.linspace(1,10,100)
    y=np.sin(x)
    subplot.plot(x,y)
    output=figure2canvas(fig)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

def figure2canvas(fig):
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    return output

if __name__ == '__main__':
    flaskapp.run(debug =True)
