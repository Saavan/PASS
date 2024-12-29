import sys
import torch
import numpy as np
from IsingSBM import CutSBM, CutBM 
from numba import njit



#testDir = 'Hamerly2/maxcut/'
testDir = 'Hamerly2019_Data/maxcut/'
solnFile = 'Hamerly2019_Data/gs_maxcut.txt'

@njit(cache=True)
def _strToArr(input_string):
    outArr = np.empty(len(input_string))
    for i in range(len(input_string)):
        outArr[i] = int(input_string[i] == '1')
    return outArr 

def testCut(model, samps, trials, samp_engine='dict'):
    outCuts = []
    for _ in range(trials):
        if samp_engine == 'dict':
            outStats, _ = model.generate_statistics(samps, collapse_samps=True, keep_samps=False)
            v = list(outStats.items())
            vals = [x[1] for x in v]
            MLE = _strToArr(v[np.argmax(vals)][0])
            outCuts.append(model.cut_value(MLE))
        elif samp_engine == 'hit':
            _, outStates = model.generate_statistics(samps, collapse_samps=False, keep_samps=True)
            tempCuts = np.array([model.cut_value(samp) for samp in outStates.transpose()])
            outCuts.append(np.amax(tempCuts))
        else:
            raise ValueError('Wrong input argument {0} for samp_engine'.format(samp_engine))
    return outCuts

def testID(N, prob_id, samps=1000, trials=10, temperature=1, samp_engine='dict', model_type='SBM'):
    fname = testDir + 'N{0:03}-id{1:02}.txt'.format(int(N), int(prob_id))
    if model_type == 'SBM':
        model = CutSBM(fname, temperature=temperature)
    else:
        model = CutBM(fname, temperature=temperature)
    with open(solnFile, 'r') as f:
        params = f.readline().split()
        Nind = params.index('n')
        idind = params.index('id')
        cutInd = params.index('cut')
        for line in f:
            params = line.split()
            if int(params[Nind]) == N and int(params[idind]) == prob_id:
                cut = int(params[cutInd])
                break
    rawCuts = np.array(testCut(model, samps, trials, samp_engine=samp_engine)) / cut
    return rawCuts



def solnCut(N, prob_id):
    with open(solnFile, 'r') as f:
        params = f.readline().split()
        Nind = params.index('n')
        idind = params.index('id')
        cutInd = params.index('cut')
        for line in f:
            params = line.split()
            if int(params[Nind]) == N and int(params[idind]) == prob_id:
                cut = int(params[cutInd])
                break
    return cut

def solnEnergy(N, prob_id):
    with open(solnFile, 'r') as f:
        params = f.readline().split()
        Nind = params.index('n')
        idind = params.index('id')
        HInd = params.index('H')
        for line in f:
            params = line.split()
            if int(params[Nind]) == N and int(params[idind]) == prob_id:
                gs_energy = int(params[HInd])
                break
    return gs_energy


if __name__ == "__main__":
    print("SBM test")
    print(testID(50, 0, samps=1000, trials=10, temperature=1, samp_engine='dict', model_type='SBM'))
    print(testID(50, 0, samps=1000, trials=10, temperature=1, samp_engine='hit', model_type='SBM'))
    print("BM test")
    print(testID(50, 0, samps=1000, trials=10, temperature=1, samp_engine='dict', model_type='BM'))
    print(testID(50, 0, samps=1000, trials=10, temperature=1, samp_engine='hit', model_type='BM'))
    import cProfile
    cProfile.run('testID(50, 0, samps=1000, trials=10, temperature=1, samp_engine="dict", model_type="SBM")')