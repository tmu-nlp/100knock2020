r"""knock28.py
28. MediaWikiマークアップの除去
27の処理に加えて，テンプレートの値からMediaWikiマークアップを可能な限り除去し，
国の基本情報を整形せよ．

[URL]
https://nlp100.github.io/ja/ch03.html#28-mediawikiマークアップの除去

[Ref]
- regex
    - https://docs.python.org/ja/3/library/re.html?#regular-expression-syntax
        - 特殊文字（special characters）
            - \number
                - 同じ番号のグループの中身にマッチします。
                - グループは 1 から始まる番号をつけられます。
- ウィキペディアのマークアップ早見表
    - https://ja.wikipedia.org/wiki/Help:%E6%97%A9%E8%A6%8B%E8%A1%A8
- ウィキペディアの「基礎情報 国」
    - https://ja.wikipedia.org/wiki/Template:基礎情報_国
- ウィキペディアの <ref>
    - https://ja.wikipedia.org/wiki/Wikipedia:出典を明記する
- ウィキペディアの言語タグ
    - https://ja.wikipedia.org/wiki/Template:Lang
        - {{lang|言語タグ|文字列}}
- ウィキペディアの Cite web
    - https://ja.wikipedia.org/wiki/Template:Cite_web
        - {{Cite web |url= |title= |accessdate=}}

[Usage]
python knock28.py
"""
import os
import re
import sys
from collections import OrderedDict, defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from kiyuna.utils.pickle import load  # noqa: E402 isort:skip
from kiyuna.utils.message import Renderer, message  # noqa: E402 isort:skip
from kiyuna.utils.message import green  # noqa: E402 isort:skip


def exec_sub(od: OrderedDict, pattern: str, repl: str) -> OrderedDict:
    res = OrderedDict()
    reg = re.compile(pattern, re.DOTALL)
    for key in od:
        res[key] = reg.sub(repl, od[key])
    return res


def remove_asterisks_in_ref(od: OrderedDict) -> OrderedDict:
    """ reshape bulleted list
    """
    reg = re.compile(r"(.+?)<ref>([^:\{]+):(.+?)</ref>", re.DOTALL)
    keys = list(od.keys())
    res = dict(od)
    for key, value in od.items():
        match = reg.search(value)
        if match:
            res[key], sub = match.group(1, 2)
            cnt = defaultdict(int)
            new_keys = []
            prv_lv = 0
            for e in match.group(3).strip().split("\n"):
                ast, txt = re.match("(\**)([^\*]+)", e).group(1, 2)
                if txt == "<br />":
                    continue
                lv = len(ast)
                if prv_lv > lv:
                    cnt[lv + 1] = 0
                cnt[lv] += 1
                new_key = sub + "".join(f"-{cnt[i]}" for i in range(1, lv + 1))
                new_keys.append(new_key)
                res[new_key] = txt
                prv_lv = lv
            for i, k in enumerate(new_keys, start=1):
                keys.insert(keys.index(key) + i, k)
    return OrderedDict((key, res[key]) for key in keys)


if __name__ == "__main__":
    pairs = {
        # remove emphasis syntax
        r"'{2,}": r"",
        # [double curly brackets] reshape lang templates
        r"{{lang\|(?P<Lang_tag>.+?)\|(?P<Text>.+?)}}": r"\g<Lang_tag>:\g<Text>",
        r"{{Cite web\|.*?title=(?P<Title>.+?)(?:\|.*?)}}": r"\g<Title>",
        r"{{en icon}}": r" (英語)",
        r"<br />{{.+}}": r"",
        # [double square brackets] replace interwiki link and extended image syntax for displayed characters
        r"\[\[.*?([^\|]+?)\]\]": r"\1",
        # [single square bracket] replace external link for displayed characters
        r"\[.*?([^ \|]+?)\]": r"\1",
        # remove footnotes
        r'<ref name=".+?" />': r"",
        r'<ref(?: name=".+?")?>(.*)</ref>': r" (\1)",
        # remove <br />
        r"<br />": r"\n",
    }

    infobox = load("infobox")

    res = remove_asterisks_in_ref(infobox)
    for pair in pairs.items():
        res = exec_sub(res, *pair)

    with Renderer("knock28") as out:
        for key in res:
            src = infobox.get(key, "該当なし")
            dst = res[key]
            if src == dst:
                out.cnt += 1
            else:
                out.result(key, (src, green(dst)))
        if infobox == res:
            message("変化なし", type="warning")
