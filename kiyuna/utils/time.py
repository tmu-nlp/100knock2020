import os
import sys
import time
from typing import Type

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from kiyuna.utils.message import message  # noqa: E402 isort:skip


class Timer(object):
    start: float
    end: float
    secs: float
    msecs: float

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self) -> Type["Timer"]:
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000
        if self.verbose:
            message(f"elapsed time = {self.msecs:f} [msec]", type="success")
