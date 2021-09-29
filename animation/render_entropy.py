import matplotlib.pyplot as plt
from animation.render import Render

class RenderEntropy(Render):
    def __init__(self, pause, belief, source, state, entropy):
        super().__init__(pause, belief, source, state)
        self.entropy_level = self.ax2.bar(0, entropy, color='pink', label='entropy')
        self.ax2.set_ylim([0,entropy])   # maximum entropy is initial entropy (entropy of the flat distrib)
        self.ax2.set_xlim(-1, 1)  # to adjust width of the bar
        self.ax2.set_xticks([])
        
    def generate_axis(self):
        self.fig, (self.ax, self.ax2) = plt.subplots(1, 2, figsize=(16,10), gridspec_kw={'width_ratios': [4, 1]})
        
    def show(self, t, obs, belief, state, entropy):
        self.update_plots(t, obs, belief, state)
        self.entropy_level[0].set_height(entropy)
        self.redraw()
        
        
        
class RenderEntropyThreshold(RenderEntropy):

    def __init__(self, pause, belief, source, state, entropy, threshold):
        super().__init__(pause, belief, source, state, entropy)
        self.ax2.axhline(threshold, label='threshold')  # show entropy threshold as an horizontal line superimposed to the entropy bar
        plt.legend()
