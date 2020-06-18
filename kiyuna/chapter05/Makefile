#
# 日本語Wikipediaの「人工知能」に関する記事からテキスト部分を抜き出したファイルが
# ai.ja.zipに収録されている． この文章をCaboChaやKNP等のツールを利用して係り受け解析を行い，
# その結果をai.ja.txt.parsedというファイルに保存せよ．
# このファイルを読み込み，以下の問に対応するプログラムを実装せよ．
#

OBJ = ai.ja

all: $(OBJ).txt.parsed

$(OBJ).zip:
	wget -qN https://nlp100.github.io/data/ai.ja.zip

$(OBJ).txt.parsed: $(OBJ).zip
	# mecab のオプションについて
	# 	-f1 なし: tree（CABOCHA_FORMAT_TREE）
	# 	-f1 あり: lattice（CABOCHA_FORMAT_LATTICE）
	# 括弧の対応を合わせるために，一部の半角記号を全角に置換
	#	↓ このような出力を回避 ↓
	# 	(    名詞,サ変接続,*,*,*,*,*
	# 	ニッポニカ    名詞,一般,*,*,*,*,*
	# 	)』    名詞,サ変接続,*,*,*,*,*
	unzip -c $< $(OBJ).txt | tail +3 | sed -e "s/(/（/g" -e "s/)/）/g" -e "s/!/！/g" | cabocha -f1 -o $@

.PHONY: clean
clean:
	-rm $(OBJ).zip $(OBJ).txt.parsed  # make で作成
	-rm out44_*.png                 # knock44.py で作成
	-rm out45*                      # knock45.py で作成

help:
	@echo "make と入力すれば $(OBJ).zip をダウンロードし，$(OBJ).txt.parsed を作成します" >&2
