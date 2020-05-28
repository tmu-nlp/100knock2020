# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

# 区切り文字がタブ(sep='\t')なので「read_table()」を用いる
# headerがないので「None」を指定
data = pd.read_table('popular-names.txt', sep='\t', header=None)

# 「sort_values()」でソートする
# 第1引数を「2」とすることで3列目を指定
# 「ascending=False」とすることで降順になる
print(data.sort_values(2, ascending=False))

# 確認
# sortコマンドで3列目を降順に並べ替える
