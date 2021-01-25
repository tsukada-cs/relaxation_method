#%%
import copy

import numpy as np

# %%
class RM:
    def __init__(self, vsets, ksize=3, weight=False):
        self.vsets = vsets
        self.ksize = ksize
        self.ci = ksize//2
        self.half = self.ci
        self.scores = np.array(list(map(lambda vectors: list(map(lambda v: [1/len(v)]*len(v), vectors)), vsets)))
        self.weight = weight
        self.wkl = np.ones([ksize,ksize])
        self.Dkl = np.zeros([ksize,ksize])
        yy, xx = np.mgrid[self.half:-(self.half+1):-1,-self.half:self.half+1]
        self.Dkl = np.sqrt(yy**2 + xx**2)
        if weight:
            self.wkl = 1/self.Dkl
            self.wkl[self.ci, self.ci] = 0
        self.wkl /= np.sum(self.wkl)

    def get_qk(self, vset, pset, i, alpha=0.5):
        wkl = copy.deepcopy(self.wkl)
        wkl[vset == None] = 0
        wkl /= np.sum(wkl)
        qk = 0
        for row in range(self.ksize):
            for col in range(self.ksize):
                if row == col == self.ci:
                    continue
                if vset[row, col] is None:
                    continue
                for idash, d_idash in enumerate(vset[row, col]):
                    if i == 0 and pset[row, col][idash] == max(pset[row, col]):
                        qk += 0.5
                        continue
                    if i == 0 or idash == 0:
                        continue
                    d_i = vset[self.ci,self.ci][i]
                    d_len_square = (d_i[0] - d_idash[0])**2 + (d_i[1] - d_idash[1])**2
                    Lkl_square = (alpha * self.Dkl[row, col])**2
                    rkl = wkl[row, col] * np.exp(-np.log(2)*d_len_square/Lkl_square)
                    pl = pset[row, col][idash]
                    qk += rkl * pl
        return qk

    def get_pk(self, vset, pset, i):
        numerator = pset[self.ci,self.ci][i] * self.get_qk(vset, pset, i)
        denominator = 0
        for j in range(len(vset[self.ci,self.ci])):
            denominator += pset[self.ci,self.ci][j] * self.get_qk(vset, pset, j)
        pk = numerator/denominator
        if i == 0 and len(pset[self.ci,self.ci]) >= 3: #TODO >=3 いらないのでは
            pk = max([pk, sorted(pset[self.ci,self.ci][1:])[-2]])
        return pk
    
    def scoreing(self, epoch=10):
        vsets = np.pad(self.vsets, self.half, mode="constant", constant_values=None)
        scores = np.pad(self.scores, self.half, mode="constant", constant_values=None)
        
        for _ in range(epoch):
            tmp_scores = scores[:]
            for row in range(self.half, vsets.shape[0]-self.half):
                for col in range(self.half, vsets.shape[1]-self.half):
                    vset = vsets[row-self.half:row+self.half+1, col-self.half:col+self.half+1]
                    pset = scores[row-self.half:row+self.half+1, col-self.half:col+self.half+1]
                    for i in range(len(vset[self.ci,self.ci])):
                        tmp_scores[row,col][i] = self.get_pk(vset, pset, i)
            self.scores = tmp_scores[self.half:-self.half, self.half:-self.half]
        return self.scores

#%%
# vsets = np.array([
#     [[None,[0,1],[2,3],[4,1]], [None,[0.4,1],[2.2,2.9],[3.7,1.3]], [None,[-1,1],[1.4,2.3],[3.2,1.5],[2.7, 1.0]]],
#     [[None,[2,1],[4,2.3],[3,1.5], [-2.7,-1.0]], [None,[2,4],[6, 2.1],[-1,1.5],[2.7,-1.0]], [None,[0,-1],[2,2],[2,4],[1,3]]],
#     [[None,[0,1],[2,5],[-1,4],[1,3]], [None,[10,2],[2,2],[2,44],[-1,3]], [None,[4,-1],[0,2],[2,5],[1,-3]]]
# ])

vsets = np.array([
    [[None,[0,1],[0,1.6]], [None,[0,1.3],[0,1.3]], [None,[0,1],[0,1.5]]],
    [[None,[0,1],[0,1.5]], [None,[0,1.1],[0,11]], [None,[0,1],[0,1.5]]],
    [[None,[0,1],[0,1.7]], [None,[0,2],[0,1.32]], [None,[0,1],[0,1.4]]],
])

# vsets = np.pad(vsets, 5, mode="reflect")

rm = RM(vsets, ksize=3, weight=True)
scores = rm.scoreing(epoch=1)
#%%
scores[1,1]
# %%
scores
# %%
