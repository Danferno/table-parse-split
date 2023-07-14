import pandas as pd
import numpy as np
import numba as nb
import pyperf
import timeit

df = pd.read_parquet('temp.pq')     # doesn't work anymore because temp.pq changed
df.loc[2, 'text'] = "CODesTEIOZezaeEZROHEZRHZ"

LOOPS = 1000

@nb.njit(nb.int32(nb.types.unicode_type))
def getLongest_nb(string):
    max_length = 0
    current_length = 0

    for char in string:
        if char.isupper():
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0

    return max_length

def getLongest_py(string):
    max_length = 0
    current_length = 0

    for char in string:
        if char.isupper():
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0

    return max_length

def fullNumba(df):
    _ = df['text'].apply(getLongest_nb)
def noNumba(df):
    _ = df['text'].apply(getLongest_py)


# warmup
assert df['text'].apply(getLongest_nb).equals(df['text'].apply(getLongest_py))

# test
numbaTime = timeit.timeit(stmt='fullNumba(df)', globals=globals(), number=LOOPS)
pythonTime = timeit.timeit(stmt='noNumba(df)', globals=globals(), number=LOOPS)

print(f'Numba: {numbaTime}\nPython: {pythonTime}')      # 1000 | Numba: 2.5, Python: 1.3