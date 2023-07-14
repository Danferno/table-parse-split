import pandas as pd
import numpy as np
import numba as nb
import pyperf
import timeit
import pickle
with open('temp.pkl', 'rb') as f:
    mask = pickle.load(f)

LOOPS = 1000

@nb.vectorize([nb.int32(nb.int32, nb.int32)])
def reset_cumsum(x, y):
    return x + y if y else 0
@nb.jit(nopython=True)
def zero_start(array):
    for idx, row in enumerate(array):
        zeros = np.where(row == 0)[0]
        if len(zeros):
            array[idx, :zeros[0]] = 0
        else:
            array[idx] = 0
    return array

def fullNumba_cumsum(mask):
    return reset_cumsum.accumulate(mask, axis=1)
def fullNumba_zerostart(cumNoText_left):
    return zero_start(cumNoText_left)
def noNumba_zerostart(cumNoText_left):
    return zero_start.py_func(cumNoText_left)

# warmup
cumNoText_left = fullNumba_cumsum(mask)
cumNoText_left_nb = fullNumba_zerostart(cumNoText_left)
cumNoText_left_py = noNumba_zerostart(cumNoText_left)

# test
numbaTime = timeit.timeit(stmt='fullNumba_cumsum(mask)', globals=globals(), number=LOOPS)
print(f'Cumsum\n\tNumba: {numbaTime}')                                  # 1000 | Numba 0.9

pythonTime = timeit.timeit(stmt='noNumba_zerostart(cumNoText_left)', globals=globals(), number=LOOPS, setup='cumNoText_left = fullNumba_cumsum(mask)')
numbaTime = timeit.timeit(stmt='fullNumba_zerostart(cumNoText_left)', globals=globals(), number=LOOPS, setup='cumNoText_left = fullNumba_cumsum(mask)')
print(f'Zerostart\n\tNumba: {numbaTime}\n\tPython: {pythonTime}')       # 1000 | Numba 0.7, Python: 2.8