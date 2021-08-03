import matplotlib.pyplot as plt
import numpy as np

class Render():
    def __init__(self, pause, belief, source, state, estimated_target):
        self.pause=pause    # time interval between animation frames
        self.fig = plt.figure(figsize=(16, 10))
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.imshow(belief, cmap='Greens')   # heatmap for the belief
        self.scat_source = self.ax.scatter(source[1], source[0], color='r', marker='*', label="real target", s=150)
        self.scat_me = self.ax.scatter(state[1], state[0], color='y', marker='o', s=5, zorder=0.1, label="visited positions")
        self.scat_mel = self.ax.scatter(state[1], state[0], color='orange', marker='o', label="current position", zorder=1, s=100)
        self.scat_obs = self.ax.scatter(None, None, color='purple', marker='o', label="observations", s=5, zorder=0.2)
        self.scat_target = self.ax.scatter(estimated_target[1], estimated_target[0], color='b', marker='x', label="estimated target", s=150)
        self.fig.colorbar(self.im, ax=self.ax, ticks=None)
        self.text = self.fig.text(0.4, 0.6, "")
        plt.show(block=False)
        
    def update_plots(self, t, obs, belief, state, estimated_target):
        self.im.autoscale()
        self.im.set_array(belief)
        self.scat_me.set_offsets(np.vstack([self.scat_me.get_offsets(), np.array([state[1],state[0]])]))
        self.scat_mel.set_offsets([state[1], state[0]])
        if obs:
            self.scat_obs.set_offsets(np.vstack([self.scat_obs.get_offsets(), np.array([state[1],state[0]])]))
        self.scat_target.set_offsets([estimated_target[1], estimated_target[0]])
        self.ax.set_title("t="+str(t))
        if t==0:
            self.ax.legend(bbox_to_anchor=(-0.2, 0.5))
        self.text.set_text("y=1" if obs else "")
    
    
    #show: display a graphical representation of the grid

    def show(self, t, obs, belief, state, estimated_target):
        self.update_plots(t, obs, belief, state, estimated_target)        
        self.fig.canvas.draw()
        if self.pause!=0:     # pause=0 is the fastest animation
            plt.pause(self.pause)
        
    def field(self, field, state, source):
        #plt.figure(figsize=(16, 10))
        plt.imshow(field)
        plt.colorbar()
        plt.show()
        # copiarci dentro self.scat_me e self.scat_source?
