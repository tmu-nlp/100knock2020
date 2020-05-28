# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

# 区切り文字がタブ(sep='\t')なので「read_table()」を用いる
# headerがないので「None」を指定
data = pd.read_table('popular-names.txt', sep='\t', header=None)

# 「data[0]」で1列目を指定し，「unique」で異なる文字列の集合を重複を除いて求める．
print(data[0].unique())

# 確認
# sortコマンドで名前順に並べ替えて，uniqコマンドで連続する同じ行を削除する．
