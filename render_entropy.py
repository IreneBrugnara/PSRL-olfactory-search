import matplotlib.pyplot as plt
import numpy as np
from render import Render

class RenderEntropy(Render):
    def __init__(self, pause, belief, source, state, entropy):
        super().__init__(pause, belief, source, state)
        self.entropy_level = self.ax2.bar(0, entropy, color='pink', label='entropy')
        self.ax2.set_ylim([0,entropy])
        self.ax2.set_xlim(-1, 1)  # to adjust width of the bar
        self.ax2.set_xticks([])
        
    def generate_axis(self):
        self.fig, (self.ax, self.ax2) = plt.subplots(1, 2, figsize=(16,10), gridspec_kw={'width_ratios': [4, 1]})
        
    def show_threshold(self, threshold):  # vorrei che andasse nella init, solo che il threshold non c'è là
        self.ax2.axhline(threshold, label='threshold')
        plt.legend()
        
    def show(self, t, obs, belief, state, entropy):
        self.update_plots(t, obs, belief, state)
        self.entropy_level[0].set_height(entropy)
        self.redraw()
        
