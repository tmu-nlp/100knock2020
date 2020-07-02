#
# 本章では，Fabio Gasparetti氏が公開しているNews Aggregator Data Setを用い，
# ニュース記事の見出しを「ビジネス」「科学技術」「エンターテイメント」「健康」の
# カテゴリに分類するタスク（カテゴリ分類）に取り組む．
#

OBJ = NewsAggregatorDataset

all: $(OBJ)

$(OBJ).zip:
	wget -qN https://archive.ics.uci.edu/ml/machine-learning-databases/00359/$@

$(OBJ): $(OBJ).zip
	mkdir -p $@
	unzip $< -d $@

.PHONY: clean
clean:
	-rm -r $(OBJ) $(OBJ).zip  # make で作成
	-rm train.txt valid.txt test.txt                        # knock50.py で作成
	-rm *.feature.txt                                       # knock51.py で作成
	-rm out58.png                                           # knock58.py で作成

help:
	@echo "make と入力すれば $(OBJ).zip をダウンロードし，$(OBJ).txt.parsed を作成します" >&2
