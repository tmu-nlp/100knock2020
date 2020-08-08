"""
99. 翻訳サーバの構築Permalink
ユーザが翻訳したい文を入力すると，その翻訳結果がウェブブラウザ上で表示されるデモシステムを構築せよ．
"""

from bottle import route, run, template, request
from datetime import datetime
import subprocess
import MeCab
import time


GPU = 3
shell = "CUDA_VISIBLE_DEVICES={GPU} PYTHONIOENCODING=utf-8 fairseq-interactive ./use_file/AT_bin --path model/model1_m/checkpoint10.pt --beam 1"

@route('/translate')
def output():
    now = datetime.now()
    return template("knock99", text_inp="", text_res="")

tagger = MeCab.Tagger("-Owakati")

@route("/translate", method="POST")
def translate():
    proc = subprocess.Popen(shell, encoding='utf-8', stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    input_text = request.forms.input_text
    input_ = tagger.parse(input_text)
    proc.stdin.write(input_)
    proc.stdin.close()
    res = proc.stdout.readlines()[-2].strip().split("\t")[-1]

    return template("knock99", text_inp=input_text, text_res=res)

run(host="localhost", port=8080, debug=True)