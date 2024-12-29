from async import BM, SBM
import numpy as np 
from numba import jit, njit



def populate_params(fname):
    f = open(fname, 'r')
    #First Line is parameters, number of vertices and number of edges
    params = f.readline()[:-1].split()
    params = list(map(float, params))
    params = list(map(int, params))
    num_vertices = int(params[0])
    weights = np.zeros((num_vertices, num_vertices), dtype=np.float64)
    bias = np.zeros(num_vertices, dtype=np.float64)
    for line in f:
        inds = line[:-1].split()
        #First two indices
        inds[0] = int(float(inds[0]))
        inds[1] = int(float(inds[1]))
        #Weight value between them (can be float or int)
        inds[2] = float(inds[2])
        weights[inds[0]-1, inds[1]-1] = inds[2]
        weights[inds[1]-1, inds[0]-1] = inds[2]
    f.close()
    return weights,bias

class IsingSBM(SBM):
    '''
    SBM for evaluating Ising Model (fully connected {-1, 1} activation}  problems
    '''
    def __init__(self, fname, temperature=1, rtime=1e-7, rate=1):
        '''
        fname is file to MaxCUT data from
        '''
        weights,bias = populate_params(fname)
        self.adj = weights.copy()

        #This inverts the weight matrix (so that cuts are incentivized)
        weights = -1 * weights
        #Going from a s = {-1, 1} to s = {0, 1}
        bias = temperature * 2 * (bias - np.dot(weights, np.ones(bias.shape[0])))
        weights = temperature * 4*weights
        super().__init__(weights, bias)
        self.rtime=rtime
        self.rate=rate

    def ising_energy(self, state):
        state2 = 2 * state - 1
        return np.matmul(np.matmul(state2, self.adj), state2)

class CutSBM(IsingSBM):
    def cut_value(self, state):
        '''
        Takes a state and returns the value of the cut created by
        partitioning through those states
        '''
        adj = self.adj
        return _cut_value_jit(adj, adj.shape[0], state)

class IsingBM(BM):
    '''
    SBM for evaluating Ising Model (fully connected {-1, 1} activation}  problems
    '''
    def __init__(self, fname, temperature=1, rtime=1e-7, rate=1):
        '''
        fname is file to MaxCUT data from
        '''
        weights,bias = populate_params(fname)
        self.adj = weights.copy()

        #This inverts the weight matrix (so that cuts are incentivized)
        weights = -1 * weights
        #Going from a s = {-1, 1} to s = {0, 1}
        bias = temperature * 2 * (bias - np.dot(weights, np.ones(bias.shape[0])))
        weights = temperature * 4*weights
        super().__init__(weights, bias)
        self.rtime=rtime
        self.rate=rate

    def ising_energy(self, state):
        state2 = 2 * state - 1
        return np.matmul(np.matmul(state2, self.adj), state2)

class CutBM(IsingBM):
    def cut_value(self, state):
        '''
        Takes a state and returns the value of the cut created by
        partitioning through those states
        '''
        adj = self.adj
        return _cut_value_jit(adj, adj.shape[0], state)

@jit(nopython=True, cache=True)
def _cut_value_jit(adj, num_visible, state):
    cut = 0
    for i in range(num_visible):
        outgoing_edges = adj[i, :]
        for j in range(i):
            if outgoing_edges[j] != 0:
                if state[i] != state[j]:
                    cut += 1
    return cut

if __name__ == "__main__":
    testDir = '/home/saavan/Box/Dropbox/Research/RBM/IsingSimulator/MaxCUT/maxcut_probs/'
    solnFile = '/home/saavan/Box/Dropbox/Research/RBM/IsingSimulator/MaxCUT/maxcut_probs/gs_maxcut.txt'
    testFile = testDir + 'N010-id00.txt'
    maxCut = 17
    N10_SBM = CutSBM(testFile, temperature=2)
    N10_BM = CutBM(testFile, temperature=2)
    SBM_dict, SBM_samps = N10_SBM.generate_statistics(100, keep_samps=True)
    BM_dict, BM_samps = N10_BM.generate_statistics(100, keep_samps=True)
    print(SBM_samps.shape)
    print(SBM_dict)
    for samp in SBM_samps.transpose():
        print(N10_SBM.cut_value(samp))