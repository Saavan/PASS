import numpy as np
from numba import jit, njit, vectorize
from numba.core import types
from numba.typed import Dict
import matplotlib.pyplot as plt
def acf(x, length=20):
     """Autocorrelation function for an input vector x. Note this normalizes with respect
     to variance, so vectors that are very slow moving will return NaN
     x - vector to calculate autocorrelation of
     length - length to calculate autocorrelation coefficient out to
      """
     return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)])


def taverage(s, t):
    avgs = np.zeros(2**s[0].shape[0])
    sprev = s[0]
    tprev = 0
    for i, j in zip(s, t):
        if np.allclose(i, sprev):
            exps = np.array([2**(len(i) - x-1) for x in range(len(i))]) 
            ind = np.sum(exps*i)
            #ind = 4*i[0] + 2*i[1] + i[2]
            avgs[ind] += j - tprev
        sprev = i
        tprev = j
    return avgs/np.sum(avgs)


class Sampler():
    def __init__(self, weights, bias):
        if not (len(weights.shape) == 2 and weights.shape[0] == weights.shape[1]):
            raise ValueError("Weights should be square matrix!")
        if not (weights.shape[0] == bias.shape[0]):
            raise ValueError("Weights and biases should have same size!")
        #Weight matrix
        self.weights = weights
        #Biases
        self.bias = bias
    def generate_statistics(self, num_samples, samp_period=1, keep_samps=False):
        pass

class SBM(Sampler):
    def __init__(self, weights, bias, rtime=1e-7, rate=1):
        super().__init__(weights, bias)
        #Base exponential rate variable 
        self.rate = rate
        #Rise time for neurons in this model
        self.rtime = rtime

    def generate_stream(self, time):
        """
        Generates a stochastic bitstream for the given boltzmann weights based on poisson process update rule
        """
        tvals, svals = _gen_stream_helper(self.weights, self.bias, self.rate, self.rtime, time)
        return (np.array(tvals), np.array(svals))
    def generate_statistics(self, num_samples, samp_period=1, collapse_samps=True, keep_samps=False):
        """
        Given a number of samples and sampling period, generate statistics for the current problem
        output is a dictionary of samples, the key being a bit string, the value is the number of appearances of the string
        if keep_samps=True, also outputs raw time series data
        samp_period sets sampling period of stochastic stream
        neuron_rate sets the base rate parameter of each of the neurons
        """
        tvals, svals = self.generate_stream(num_samples*samp_period)
        samp_step = np.arange(0, num_samples*samp_period, samp_period)
        samp_vals = []
        for i in range(svals.shape[1]):
            samp_vals.append(np.interp(samp_step, tvals, svals[:, i]))

        samp_vals = np.array(samp_vals, dtype=np.uint8)

        outDict = {}
        outVals = np.array([])
        if collapse_samps:
            outDict = _collapse_samps(samp_vals)
        if keep_samps:
            outVals = samp_vals
        return outDict, outVals



class BM(Sampler):
    def __init__(self, weights, bias):
        super().__init__(weights, bias)

    def generate_statistics(self, num_samples, samp_period=1, collapse_samps=True, keep_samps=False):
        '''
        Baseline boltzmann machine sampler
        '''
        samps = np.array(_BM_helper(self.weights, self.bias, num_samples), dtype=np.uint8)
        outDict = {}
        outVals = np.array([])
        if collapse_samps:
            outDict = _collapse_samps(samps)
        if keep_samps:
            outVals = samps
        return outDict, outVals
        
@njit(cache=True)
def _gen_stream_helper(weights, bias, rate, rtime, time):
    #initializes states to random values
    s = np.array([int(np.random.random() > 0.5) for _ in range(bias.shape[0])], dtype=np.float64)
    #probability vecors for each state
    p = _sigmoid(np.dot(weights,s)+bias)
    tvals = [0.]
    svals = []
    svals.append(np.copy(s))
    t = 0
    dt = 0
    while t < time:
        p = _sigmoid(np.dot(weights, s) + bias)
        #Setting rate parameters for each neuron
        lamb = np.where(s==1, rate*(1 - p), rate*p)
        rands = np.random.random(s.shape)
        #This is a the same as a poisson clock with different rates depending on neuron
        times = np.log(rands) / (-1 * lamb)
        #The time change is the first flipped neuron
        dt_ind = np.argmin(times)
        dt = times[dt_ind]
        #Saving the state right before we flip
        tvals.append(t + dt - rtime)
        svals.append(np.copy(s))
        #Changing the state of the neuron that flipped
        s[dt_ind] = 1 - s[dt_ind]
        #Adding the changed state to the vector
        tvals.append(t + dt)
        svals.append(np.copy(s))
        t = t + dt
    return tvals, svals

@njit(cache=True)
def _BM_helper(weights, bias, num_samples):
    #array holding output samples
    out_samps = np.empty((bias.shape[0], num_samples), dtype=np.float64)
    #initializes states to random values
    s = np.array([int(np.random.random() > 0.5) for _ in range(bias.shape[0])], dtype=np.float64)
    out_samps[:, 0] = np.copy(s)
    for i in range(num_samples):
        ind = np.random.randint(bias.shape[0])
        p = _sigmoid(np.dot(weights[ind], s) + bias[ind])
        s[ind] = int(p > np.random.random())
        out_samps[:, i] = np.copy(s)
    return out_samps

@vectorize(["float64(float64)", "float32(float32)", "float32(uint8)"], cache=True)
def _sigmoid(x):
    return 1/(1 + np.exp(-1*x))

@njit(cache=True)
def _collapse_key(byte_arr):
    out_string = ''
    for element in byte_arr:
        out_string += str(element)
    return out_string

@njit(cache=True)
def _collapse_samps(samp_vals):
    out_dict = dict()
    for i in range(samp_vals.shape[1]):
        dict_key = _collapse_key(samp_vals[:, i])
        if dict_key in out_dict:
            out_dict[dict_key] += 1
        else:
            out_dict[dict_key] = 1
    return out_dict



if __name__ == '__main__':
    #matrices for s = {-1, 1}, sigm = tanh
    J = np.array([[0., -1., 2.], [-1., 0., 2.], [2., 2., 0.]])
    b = np.array([1., 1., -2.])
    #Transformation to s = {0, 1} sigm = regular
    b2 = 2 * (b - np.dot(J, np.ones(3)))
    J2 = 4*J
    print(b2, J2)
    time = 10000
    andSBM = SBM(J2, b2)
    t, s = andSBM.generate_stream(time)
    # plt.subplot(311)
    # plt.plot(t, s[:, 0])
    # plt.subplot(312)
    # plt.plot(t, s[:, 1])
    # plt.subplot(313)
    # plt.plot(t, s[:, 2])
    # plt.show()
    # avgs = np.zeros(8)
    sprev = s[0]
    tprev = 0
    print('SBM stats: ' + str(andSBM.generate_statistics(10000)[0]))
    
    andBM = BM(J2, b2)
    print('BM stats: ' + str(andBM.generate_statistics(10000)[0]))
    import cProfile
    cProfile.run('andSBM.generate_statistics(10000)[0]')

