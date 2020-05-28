# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

# 区切り文字がタブ(sep='\t')なので「read_table()」を用いる
# headerがないので「None」を指定
data = pd.read_table('popular-names.txt', sep='\t', header=None)

# 結果
print("行数は，" + str(len(data)) + "行です．")

# 確認
# wcコマンドの代わりとして「find /c /v "" popular-names.txt」を入力．
