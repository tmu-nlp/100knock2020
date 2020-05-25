#
# popular-names.txtは，アメリカで生まれた赤ちゃんの「名前」「性別」「人数」「年」を
# タブ区切り形式で格納したファイルである．
#
.PHONY: clean

OBJ = popular-names.txt

all: $(OBJ)

$(OBJ):
	wget -qN https://nlp100.github.io/data/popular-names.txt

clean:
	-rm $(OBJ)  # make で作成
	-rm col1.txt col2.txt  # knock12.py で作成
	-rm col3.txt col4.txt  # knock12.py で作成
	-rm out13a out13b      # knock13.py で作成
	-rm out16*             # knock16.py で作成
	-rm out18*             # knock18.py で作成

help:
	@echo "make と入力すれば popular-names.txt がダウンロードできます" >&2
