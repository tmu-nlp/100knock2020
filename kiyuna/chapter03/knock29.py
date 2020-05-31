r"""knock29.py
29. 国旗画像のURLを取得する
テンプレートの内容を利用し，国旗画像のURLを取得せよ．
（ヒント: MediaWiki APIのimageinfoを呼び出して，ファイル参照をURLに変換すればよい）

[URL]
https://nlp100.github.io/ja/ch03.html#29-国旗画像のurlを取得する

[Ref]
- MediaWiki API
    - https://www.mediawiki.org/wiki/API:Imageinfo/ja

[Usage]
python knock29.py
"""
import json
import os
import pprint
import sys
import urllib.parse
import urllib.request
import webbrowser
from typing import Union

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip


url = "https://www.mediawiki.org/w/api.php"


def make_payload(filename: str) -> dict:
    return {
        "action": "query",
        "titles": "File:" + filename,
        "prop": "imageinfo",
        "format": "json",
        "iiprop": "url",
    }


def fetch_url_of_img_with_urllib(filename: str) -> dict:
    message("fetch:", f"url of `{filename}`", type="status")
    url_ = url + "?%s" % urllib.parse.urlencode(make_payload(filename))
    with urllib.request.urlopen(url_) as f:
        return json.loads(f.read().decode("utf-8"))


def fetch_url_of_img_with_requests(filename: str) -> dict:
    message("fetch:", f"url of `{filename}`", type="status")
    with requests.Session() as s:
        return s.get(url=url, params=make_payload(filename)).json()


def save_file_from_url(url: str, filename: str) -> None:
    message("save :", filename, type="status")
    with urllib.request.urlopen(url) as f_in, open(filename, "wb") as f_out:
        f_out.write(f_in.read())


def render_html(path_img, path_html="out29.html"):
    message("save :", path_html, type="status")
    contents = (
        "<!DOCTYPE html><html>"
        "<head><title>knock29</title></head>"
        '<body><img src="%s" width="128"/></body>'
        "</html>" % path_img
    )
    with open(path_html, "w") as f:
        f.write(contents)

    message("open :", path_html, type="status")
    webbrowser.open(path_html)


def flatten_json(json_data: dict) -> dict:
    ret_dict = {}
    for k, v in json_data.items():
        if isinstance(v, list):
            for e in v:
                ret_dict.update(flatten_json(e))
        elif isinstance(v, dict):
            ret_dict.update(flatten_json(v))
        else:
            ret_dict[k] = v
    return ret_dict


def flatten__json(json_data: dict) -> dict:
    res = {}

    def flatten(x: Union[dict, list, str], names=[]) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                flatten(v, names + [k])
        elif isinstance(x, list):
            for i, v in enumerate(x):
                flatten(v, names + [str(i)])
        else:
            res["__".join(names)] = x

    flatten(json_data)
    return res


if __name__ == "__main__":

    filename = load("infobox")["国旗画像"]

    # data = fetch_url_of_img_with_urllib(filename)
    data = fetch_url_of_img_with_requests(filename)
    # pprint.pprint(data, stream=sys.stderr)

    # url = data["query"]["pages"]["-1"]["imageinfo"][0]["url"]
    # url = flatten_json(data)["url"]
    # url = flatten__json(data)["query__pages__-1__imageinfo__0__url"]

    page: dict = next(iter(data["query"]["pages"].values()))
    image_info: dict = page["imageinfo"][0]
    url: str = image_info["url"]

    save_file_from_url(url, filename)

    render_html(filename)
