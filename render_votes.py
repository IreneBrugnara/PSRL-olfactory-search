import matplotlib.pyplot as plt

class RenderVotes():
    def __init__(self, votes, actions):
        self.ax_v = plt.gcf().add_subplot(1,2,2)  # al posto di plt.gcf() dovrei usare Gridworld.fig 
        self.bars = self.ax_v.bar(range(len(actions)), votes, width=0.5, color='b')
        #self.ax_v.set_xticklabels(actions)
        plt.xticks(range(len(actions)), actions)
        plt.ylim([0,1])
        
    def update(self, votes, chosen):
        for i, (bar, vote) in enumerate(zip(self.bars, votes)):
            bar.set_height(vote)
            color='dodgerblue' if i==chosen else 'lightskyblue'
            bar.set_color(color)
        plt.draw()   # non funziona ahimè, lo show delle barre è in ritardo di uno step rispetto allo show della belief


'''
class RenderVotes(Render):
    def __init__(self, pause, belief, source, state, estimated_target, votes):
        super().__init__(pause, belief, source, state, estimated_target))
        self.ax_v = self.fig.add_subplot(112)
        self.bars = self.ax_v.bar(range(4), votes)
        
        
    def update_plots(self, t, obs, belief, state, estimated_target, votes):
        super().update_plots(t, obs, belief, state, estimated_target)
        for bar, vote in zip(self.bars, votes):
            bar.set_height(vote)
'''
