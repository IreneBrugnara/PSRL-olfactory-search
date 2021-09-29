import matplotlib.pyplot as plt
import numpy as np

class Render():
    def __init__(self, pause, belief, source, state):
        self.pause=pause    # time interval (in seconds) between animation frames
        self.generate_axis()
        self.setup_plots(belief, source, state)
        plt.show(block=False)
        
    def generate_axis(self):
        self.fig, self.ax = plt.subplots(figsize=(12,10))
        
    def setup_plots(self, belief, source, state):
        self.im = self.ax.imshow(belief, cmap='Greens')   # heatmap for the belief
        self.scat_source = self.ax.scatter(source[1], source[0], color='r', marker='*', label="real target", s=150)
        self.scat_me = self.ax.scatter(state[1], state[0], color='y', marker='o', s=5, zorder=0.1, label="visited positions")
        #self.scat_mel = self.ax.scatter(state[1], state[0], color='orange', marker='o', label="current position", zorder=1, s=100)
        self.scat_obs = self.ax.scatter(None, None, color='purple', marker='o', label="observations", s=5, zorder=0.2)
        self.fig.colorbar(self.im, ax=self.ax, ticks=None)
        self.text = self.ax.text(0.1, 0.9, "", transform = self.ax.transAxes)
        self.ax.legend(loc=(-0.45,0.5))
        
    def update_plots(self, t, obs, belief, state):
        self.im.autoscale()
        self.im.set_array(belief)
        self.scat_me.set_offsets(np.vstack([self.scat_me.get_offsets(), np.array([state[1],state[0]])]))
        #self.scat_mel.set_offsets([state[1], state[0]])
        if obs:
            self.scat_obs.set_offsets(np.vstack([self.scat_obs.get_offsets(), np.array([state[1],state[0]])]))
        self.ax.set_title("t="+str(t))
        #self.text.set_text("y=1" if obs else "")
    
    
    #show: display a graphical representation of the grid

    def show(self, t, obs, belief, state):
        self.update_plots(t, obs, belief, state)
        self.redraw()        
        
    def redraw(self):
        self.fig.canvas.draw()
        if self.pause!=0:     # pause=0 is the fastest animation
            plt.pause(self.pause)
        
    def field(self, field, state, source):
        #plt.figure(figsize=(16, 10))
        plt.imshow(field, cmap='Greys_r')
        plt.colorbar()
        plt.show()
        # copiarci dentro self.scat_me e self.scat_source?
