import os
import sys
from typing import IO, Any, Callable, Optional, Tuple, Type, TypeVar, Union

import colorama
from colorama import Back, Fore, Style

Stringifiable = TypeVar("Stringifiable")

colorama.init()
prefix = {
    "emit": "",
    "status": "[" + Fore.MAGENTA + "x" + Style.RESET_ALL + "] ",
    "success": "[" + Fore.GREEN + Style.BRIGHT + "+" + Style.RESET_ALL + "] ",
    "failure": "[" + Fore.RED + Style.BRIGHT + "-" + Style.RESET_ALL + "] ",
    "debug": "[" + Fore.RED + Style.BRIGHT + "DEBUG" + Style.RESET_ALL + "] ",
    "info": "[" + Fore.BLUE + Style.BRIGHT + "*" + Style.RESET_ALL + "] ",
    "warning": "[" + Fore.YELLOW + Style.BRIGHT + "!" + Style.RESET_ALL + "] ",
    "error": "[" + Fore.RED + "ERROR" + Style.RESET_ALL + "] ",
    "exception": "[" + Fore.RED + "ERROR" + Style.RESET_ALL + "] ",
    "critical": "[" + Fore.RED + "CRITICAL" + Style.RESET_ALL + "] ",
}
red: Callable[[object], str] = lambda obj: Fore.RED + str(obj) + Fore.RESET
green: Callable[[object], str] = lambda obj: Fore.GREEN + str(obj) + Fore.RESET
yellow: Callable[[object], str] = (
    lambda obj: Fore.YELLOW + str(obj) + Fore.RESET
)
lightgreen: Callable[[object], str] = (
    lambda obj: Fore.LIGHTGREEN_EX + str(obj) + Fore.RESET
)
back_blue: Callable[[object], str] = (
    lambda obj: Back.BLUE + str(obj) + Back.RESET
)
bold: Callable[[object], str] = (
    lambda obj: Style.BRIGHT + str(obj) + Style.RESET_ALL
)
underlined: Callable[[object], str] = (
    lambda obj: colorama.ansi.code_to_chars(4) + str(obj) + Style.RESET_ALL
)


def trunc(msg: str) -> str:
    if len(msg) <= 79:
        return msg
    else:
        return msg[:63] + " ...(truncated) "


def message(
    *text: Optional[Tuple[Stringifiable]],
    CR: bool = False,
    type: str = "emit",
    file: IO[str] = sys.stderr,
) -> None:
    if isinstance(text, tuple):
        text = " ".join(map(str, text))
    elif text is None:
        text = ""
    text = prefix[type] + str(text)
    text = "\r" + text if CR else text + "\n"
    file.write(text)


class Renderer(object):
    def __init__(self, title: str) -> None:
        self.title: str = title
        self.cnt = 1

    def __enter__(self) -> Type["Renderer"]:
        message(underlined(self.title), type="success")
        return self

    def __exit__(self, *args) -> None:
        message()

    def header(self, head: str = "") -> None:
        message(bold(f"{self.cnt:2d}. {head}"), type="info")
        self.cnt += 1

    def result(self, head: str, body: Union[Tuple[object], object]) -> None:
        self.header(head)
        if isinstance(body, tuple):
            for e in body:
                message(e)
        else:
            message(body)
