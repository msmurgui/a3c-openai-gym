import matplotlib.pyplot as plt
import numpy as np
import time


class LivePlotter:
    def __init__(self, identifier=''):
        self.x_vec = np.linspace(0, 1, 300+1)[0:-1]
        self.y_vec = np.random.randn(len(self.x_vec))
        self.line1 = []
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        self.line1, = ax.plot(self.x_vec, self.y_vec, '-o', alpha=0.8)
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # use ggplot style for more sophisticated visuals
    #plt.style.use('ggplot')

    def add(self, y1_data,  pause_time=0.1):
        self.y_vec[-1] = y1_data
        # after the figure, axis, and line are created, we only need to update the y-data
        self.line1.set_ydata(self.y_vec)
        # adjust limits if new data goes beyond bounds
        if np.min(self.y_vec) <= self.line1.axes.get_ylim()[0] or np.max(self.y_vec) >= self.line1.axes.get_ylim()[1]:
            plt.ylim([np.min(self.y_vec)-np.std(self.y_vec),
                      np.max(self.y_vec)+np.std(self.y_vec)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)

        # return line so we can update it again in the next iteration
        #return line1
        self.y_vec = np.append(self.y_vec[1:], 0.0)


if __name__ == '__main__':
    plotter = LivePlotter()
    while(True):
        plotter.add(np.random.uniform())
