import os
import sys
from typing import Type

import dill

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from kiyuna.utils.message import message, trunc  # noqa: E402 isort:skip

ROOT = os.path.join(os.path.dirname(__file__), "..")


def get_path(file_name: str) -> str:
    return os.path.join(ROOT, f"pickles/{file_name}.pkl")


class SaveHelper(object):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> Type["SaveHelper"]:
        message("saving:", self.name, CR=True, type="status")
        return self

    def __exit__(self, *args) -> None:
        message("saved :", self.name, "\n", CR=True, type="success")


def dump(obj: object, file_name: str) -> None:
    with open(get_path(file_name), "wb") as f_out:
        dill.dump(obj, f_out)
    message("saved :", trunc(repr(obj)), type="success")


def load(file_name: str) -> object:
    with open(get_path(file_name), "rb") as f_in:
        obj = dill.load(f_in)
    message("loaded:", trunc(repr(obj)), type="success")
    return obj
