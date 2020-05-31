#
# Wikipediaの記事を以下のフォーマットで書き出したファイルjawiki-country.json.gzがある．
# 	- 1行に1記事の情報がJSON形式で格納される
# 	- 各行には記事名が”title”キーに，記事本文が”text”キーの辞書オブジェクトに格納され，そのオブジェクトがJSON形式で書き出される
# 	- ファイル全体はgzipで圧縮される
#
.PHONY: clean

OBJ = jawiki-country.json.gz

all: $(OBJ)

$(OBJ):
	wget -qN https://nlp100.github.io/data/jawiki-country.json.gz

clean:
	-rm $(OBJ)            # make で作成
	-rm "Flag of the United Kingdom.svg"  # knock29.py で作成
	-rm out29.html                        # knock29.py で作成

help:
	@echo "make と入力すれば jawiki-country.json.gz がダウンロードできます" >&2
