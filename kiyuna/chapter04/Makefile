#
# 夏目漱石の小説『吾輩は猫である』の文章（neko.txt）をMeCabを使って形態素解析し，
# その結果をneko.txt.mecabというファイルに保存せよ．
#

OBJ = neko.txt

all: $(OBJ).mecab

$(OBJ):
	wget -qN https://nlp100.github.io/data/neko.txt

$(OBJ).mecab: $(OBJ)
	mecab $< -o $@

.PHONY: clean
clean:
	-rm $(OBJ) $(OBJ).mecab  # make で作成
	-rm out36.png                # knock36.py で作成
	-rm out38.png                # knock38.py で作成
	-rm out39.png                # knock39.py で作成

help:
	@echo "make と入力すれば $(OBJ) をダウンロードし，$(OBJ).mecab を作成します" >&2
