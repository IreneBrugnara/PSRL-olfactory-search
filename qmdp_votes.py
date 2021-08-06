import matplotlib.pyplot as plt

class QmdpVotes():
    def __init__(self, votes, actions):
        self.ax_v = plt.gcf().add_subplot(1,2,2)
        self.bars = self.ax_v.bar(range(len(actions)), votes, width=0.5, color='b')
        #self.ax_v.set_xticklabels(actions)
        plt.xticks(range(len(actions)), actions)
        plt.title("values")
        plt.ylim([0,1])
        
    def update(self, votes, chosen):
        for i, (bar, vote) in enumerate(zip(self.bars, votes)):
            bar.set_height(vote)
            color='dodgerblue' if i==chosen else 'lightskyblue'
            bar.set_color(color)
            
    # questa classe non serve, è identica a RenderVotes
