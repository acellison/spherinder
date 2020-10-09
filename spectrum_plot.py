import numpy as np
import matplotlib.pyplot as plt

class SpectrumPlot():
    def __init__(self, evalues, callback):
        self._evalues = evalues
        self._callback = callback
        self._fig, self._ax = None, None

    @property
    def evalues(self):
        return self._evalues

    @property
    def fig(self):
        return self._fig

    @property
    def ax(self):
        return self._ax

    @property
    def callback(self):
        return self._callback

    def plot(self, *args, **kwargs):
        fig, ax = plt.subplots()
        self._fig, self._ax = fig, ax
        self.ax.plot(self.evalues.real, self.evalues.imag, '.', markersize=2, picker=5, *args, **kwargs)
        self.ax.grid()
        self.fig.canvas.mpl_connect('pick_event', self.onpick)

    def onpick(self, event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        point = tuple(zip(xdata[ind], ydata[ind]))[0]
        evalue = point[0] + 1j*point[1]
        print('onpick eigenvalue: ', evalue)

        index = np.argmin(np.abs(self.evalues-evalue))
        self.callback(index)
        return True

