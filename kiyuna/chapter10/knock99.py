"""
99. 翻訳サーバの構築
ユーザが翻訳したい文を入力すると，
その翻訳結果がウェブブラウザ上で表示されるデモシステムを構築せよ．

[Ref]
- https://ymgsapo.com/2019/09/24/flask-board-app/

[Usage]
python knock99.py 7
"""
import subprocess
import sys

import MeCab
from flask import Flask, redirect, render_template, request, session

app = Flask(__name__)
app.secret_key = "secret key"

user_data = {}
user_message = []

GPU = sys.argv[1]
EXP_NAME = "KFTT.bpe.ja-en"
BPE_CODE = "data/kftt-data-1.0/data/bpe/code"

cmd = f"CUDA_VISIBLE_DEVICES={GPU} python fairseq/fairseq_cli/interactive.py"
cmd += f" --path checkpoints/{EXP_NAME}/checkpoint_best.pt"
cmd += f" --remove-bpe --bpe=subword_nmt --bpe-codes {BPE_CODE}"
cmd += f" data-bin/{EXP_NAME}"

tagger = MeCab.Tagger("-Owakati")

cnt = 0


def translate(line):
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, encoding="utf8"
    )
    ja = tagger.parse(line)
    proc.stdin.write(ja)
    proc.stdin.close()
    res = proc.stdout.readlines()[-3].split("\t")[-1]
    return res


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", header="99. 翻訳サーバの構築", history=user_message,)


@app.route("/", methods=["POST"])
def index_post():
    global cnt
    cnt += 1
    pm = request.form["history"]
    user_message.append({"no": cnt, "src": pm, "hyp": translate(pm)})
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5555, threaded=True)
