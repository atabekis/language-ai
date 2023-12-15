import multiprocessing as mp
import time
from tqdm import *
import tqdm


def _foo(my_number):
    square = my_number * my_number
    time.sleep(0.2)
    return square


if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count())
    inputs = range(100)  # Change this to your actual inputs

    results = []
    for result in tqdm.tqdm(pool.imap_unordered(_foo, inputs), total=len(inputs)):
        results.append(result)
    print(results)
