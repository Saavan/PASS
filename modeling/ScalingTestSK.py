import torch
import time
import sys
import logging
import logging.handlers
import pickle
import numpy as np
import testSK
import signal
import multiprocessing as mp
from multiprocessing import Pool
import psutil

#Can't stop won't stop
signal.signal(signal.SIGHUP, signal.SIG_IGN)


samps = np.logspace(1, 4, num=16).astype(int)
trials = 100
n = np.arange(10, 160, 10)
probid = np.arange(0, 10)
temperature = np.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
model='BM'
engine='hit'
solns = []
njobs = 12


def test_handler(args):

    pid = mp.current_process().pid

    py = psutil.Process(pid)
    msg = "Starting: N={0}, prob_id={1}, samps={2}, trials={3}, temperature={4}, model_type={5}, samp_engine={6}, pid={7}".format(*args, pid)
    #Put the message into the shared queue
    test_handler.q.put(msg)

    out =  testSK.testID(N=args[0], prob_id=args[1], samps=args[2], trials=args[3], temperature=args[4],
                model_type=args[5], samp_engine=args[6])

    msg = "Done: N={0}, prob_id={1}, samps={2}, trials={3}, temperature={4}, model_type={5}, samp_engine={6}, perf={7:.3f}, pid={8}".format(*args, np.median(out), pid)
    test_handler.q.put(msg)


    with test_handler.counter.get_lock():
        test_handler.counter.value += 1

    test_handler.q.put("{:.2%} done".format((test_handler.counter.value)/test_handler.computations))

    return out



def test_init(q, counter, computations):
    """Initialization function to give each pool function access to the same
    message passing queue.
    """
    test_handler.q = q
    test_handler.counter = counter
    test_handler.computations = computations


def logger_thread(q, log_name, counter):
    """Thread that keeps track of messages from various threads and dumps to
    a common log file.
    """
    logging.basicConfig(level=logging.DEBUG, filename=log_name, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    while True:
        msg = q.get()
        #A None message signals that we are done
        if msg is None:
            break
        logger = logging.getLogger()
        logger.info(msg)



if __name__ == "__main__":


    #Defaulting to forking original process
    mp.set_start_method('fork')

    #Starting the Queue for message passing
    q = mp.Queue()
    counter = mp.Value('i', 0)

    fname = 'ScalingTest_SK_{0}_{1}_{2}'.format(time.strftime("%y%m%d", time.localtime()), model, engine)
    log_name = filename='logs/' + fname + '.log'

    print("Logging to: ", log_name)

    #Starting up logging process
    lp = mp.Process(target=logger_thread, args=(q, log_name, counter))
    lp.start()

    work = []

    for temp in temperature:
        for samp in samps:
            for size in n:
                for index in probid:
                    vars = (size, index, samp, trials, temp, model, engine)
                    work.append(vars)


    p = Pool(njobs, test_init, [q, counter, len(work)], 8)


    t0 = time.time()
    #Actually do the computation
    try:
        out = p.map(test_handler, work)
    except Exception as e:
        q.put(str(e))
        q.put('PARALLELISM FAILED')

    t1 = time.time()
    q.put("Time taken Parallel:{0} with {1} Jobs".format(t1 - t0, njobs))
    print("Done with computation!")

    p.close()
    p.join()

    #Close up the message logging file
    q.put(None)
    lp.join()

    #Making the outputs reasonably shaped
    out = np.array(out).reshape(len(temperature), len(samps), len(n), len(probid), -1)

    with open('outputs/' + fname + '.p', 'wb') as f:
        out_dict = {'samps':samps, 'n':n,
                'prob_id':probid, 'solns':out, 'temperature':temperature, 'trials':trials}
        pickle.dump(out_dict, f)
