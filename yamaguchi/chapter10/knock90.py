# 前処理
import MeCab
tagger = MeCab.Tagger("-Owakati")

# ファイルを開く
file = open("kyoto-test.en", "r", encoding="utf-8")
# 内容を全て読み込む
contents = file.read()

result = tagger.parse(contents)

print(result)
