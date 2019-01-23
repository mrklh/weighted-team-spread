from plot import Plot


class PlotScreen:
    def __init__(self):
        self.plots = []

    def add_plot(self, *args, **kwargs):
        plot = Plot(*args, **kwargs)
        self.plots.append(plot.get_plot())

    def get_plots(self):
        return self.plots

    def set_all_data(self, x, y):
        for plot in self.plots:
            plot.set_data(x, y)

    def set_data_by_index(self, t, i, y):
        self.plots[i].set_xdata(range(1, t/10+1))
        self.plots[i].set_ydata(y)

    def initalize_plots(self):
        pass