import multiprocessing as mp
import numpy as np

import time

work = np.array([["A", 5], ["B", 2], ["C", 1], ["D", 3]])


def work_log(work_data):
    print(" Process %s waiting %s seconds" % (work_data[0], work_data[1]))
    time.sleep(int(work_data[1]))
    print(" Process %s Finished." % work_data[0])


def pool_handler():
    p = mp.Pool(mp.cpu_count())
    p.map(work_log, work)


scores = np.zeros(3, int)
print(scores)