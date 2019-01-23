class Plot:
    def __init__(self, ax, x, y, style, label):
        self.plot, = ax.plot(x, y, style, label=label)

    def get_plot(self):
        return  self.plot

    def set_data(self, x, y):
        self.plot.set_data(x, y)

    def set_pos_data(self, x, y):
        self.plot.set_xdata(x)
        self.plot.set_ydata(y)
