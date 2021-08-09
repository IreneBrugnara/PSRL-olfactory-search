import matplotlib.pyplot as plt
import numpy as np
from render import Render

class RenderVotes(Render):
    def __init__(self, pause, belief, source, state):   # in realtà potrei passare anche actions e votes....
        super().__init__(pause, belief, source, state)
        self.bars = self.ax_v.bar(range(5), np.zeros(5), width=0.5, color='b')
        actions = ["still", "right", "left", "down", "up"]
        self.ax_v.set_xticks(range(5))
        self.ax_v.set_xticklabels(actions)
        self.ax_v.set_ylim([0,1])
        
    def generate_axis(self):
        self.fig, (self.ax, self.ax_v) = plt.subplots(1, 2, figsize=(16,10))
        
    def show(self, t, obs, belief, state, votes, chosen):
        self.update_plots(t, obs, belief, state)
        self.update_bars(votes, chosen)
        self.redraw()
        
    def update_bars(self, votes, chosen):
        for i, (bar, vote) in enumerate(zip(self.bars, votes)):
            bar.set_height(vote)
            color='dodgerblue' if i==chosen else 'lightskyblue'
            bar.set_color(color)

