from time import gmtime, sleep, strftime
from typing import Any, List, Sequence, Tuple


def n_gram(seq: Sequence[Any], n: int) -> List[Tuple[Any]]:
    return list(zip(*[seq[i:] for i in range(n)]))


def print_time(func):
    def wrapper(*args, **kwargs):
        print("in  |", strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
        res = func(*args, **kwargs)
        print("res |", res)
        res[1]["return"] = 0
        print("out |", strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
        return res

    return wrapper


@print_time
def my_func(*args, **kwargs):
    sleep(1)
    return args, kwargs


if __name__ == "__main__":
    res = my_func("NLP", name="kyuna", ans=42)
    print("res |", res)
