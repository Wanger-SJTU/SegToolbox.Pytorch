
import torch
import numpy as np
import visdom
from PIL import Image
try:
    range = xrange
except NameError:
    range = range

class Visualizer(object):
    def __init__(self, envs='main',host='http://localhost', port_idx=8097):
        self.viz = visdom.Visdom(server=host, port=port_idx,env=envs)
        self.wins=[]
        self.wins_by_name={}
        self.num_wins = 0

    def show_visuals(self, data, epoch, step):

        if not isinstance(data, dict):
            raise ValueError('input parameters should be dict')
        self._set_wins(data)
        assert len(self.wins) >= len(data.keys())
        
        for key in data.keys():
            assert isinstance(key, str)
            idx = self.wins_by_name[key]
            title = key+': (epoch: %d, step: %d)' % (epoch, step)
            if not isinstance(data[key], list):
                if isinstance(data[key], Image.Image):
                    data[key] = np.array(data[key].convert('RGB')).transpose(2,0,1)
                self.viz.image(data[key], win=self.wins[idx], opts=dict(title=title)) 
            else:
                x = np.arange(1, len(data[key]) + 1, 1)
                self.viz.line(np.array(data[key]), x, win= self.wins[idx], opts=dict(title=title))

    def _set_wins(self, data):
        for key in data.keys():
            if not key in self.wins_by_name.keys():
                self.wins_by_name.update({key:self.num_wins})
                self.wins.append(self.viz.image(torch.zeros(3, 300, 300)))
                self.num_wins += 1

if __name__ == '__main__':
    a = Visualizer()
    b = {'a':[1,2], 'b':[12,3,4,5,6]}
    print(b.keys())
    a.show_visuals(b,epoch=0,step=10)
    b = {'a':[1,2], 'd':[12,3,4,5,6]}
    a.show_visuals(b,epoch=0,step=10)
    b = {'a':[1,2], 'd':[12,3,4,5,6,10,12,35,6,7]}
    a.show_visuals(b,epoch=0,step=10)
