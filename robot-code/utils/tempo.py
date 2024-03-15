from functools import wraps
import time
import logging


logging.basicConfig(filename='timestamps.log', encoding='utf-8',
                    level=logging.DEBUG, 
                    format="%(message)s")


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logging.debug(f'{func.__name__}\t{total_time:.6f}')
        # print(f'{total_time:.6f} s {func.__name__}')
        return result
    return timeit_wrapper
