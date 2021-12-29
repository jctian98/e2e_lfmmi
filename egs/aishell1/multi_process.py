#'from multiprocessing import Pool 
#'import time
#'
#'def func(i):
#'    print(f"try to sleep {i} sec", flush=True)
#'    time.sleep(i)
#'    print(f"finish to sleep {i} sec", flush=True)
#'    return i
#'
#'p = Pool(4)
#'res = []
#'for i in range(4):
#'    res.append(p.apply_async(func, (p)))
#'p.close()
#'p.join()
#'
#'for r in res:
#'    print(r.get())

import multiprocessing
import time
import random
import sys

# print 'Testing callback:'
def mul(a, b):
    print("a")
    time.sleep(0.5*random.random())
    return a * b

def pow3(x):
    r = 0.5*random.random()
    print(f"{time.time()}: pow3 called with input {x} | sleep {r}")
    time.sleep(r)
    return x ** 3

def err_callback(x):
    print(f"bb")

if __name__ == '__main__':

    PROCESSES = 4
    print('Creating pool with %d processes\n' % PROCESSES)
    pool = multiprocessing.Pool(PROCESSES)

    A = []
    B = [56, 0, 1, 8, 27, 64, 125, 216, 343, 512, 729]

    r = pool.apply_async(mul, (7, 8), callback=A.append)
    r.wait()
    
    print(A)
    print(B)

    pow3(2)
    r = pool.map_async(pow3, range(10), callback=A.extend, error_callback=err_callback)
    r.wait()
    
    print(A)
    print(B)

