import matplotlib.pyplot as plt
from animation.render import Render

class RenderWithEstTarget(Render):
    def __init__(self, pause, belief, source, state):
        super().__init__(pause, belief, source, state)
        self.scat_target = self.ax.scatter(None, None, color='b', marker='x', label="estimated target", s=150)
        self.ax.legend(loc=(-0.45,0.5))

    def update_plots(self, t, obs, belief, state, estimated_target):
        super().update_plots(t, obs, belief, state)
        self.scat_target.set_offsets([estimated_target[1], estimated_target[0]])
        
    def show(self, t, obs, belief, state, estimated_target):
        self.update_plots(t, obs, belief, state, estimated_target)
        self.redraw()
