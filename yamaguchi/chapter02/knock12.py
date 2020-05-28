# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

# 区切り文字がタブ(sep='\t')なので「read_table()」を用いる
# headerがないので「None」を指定
data = pd.read_table('popular-names.txt', sep='\t', header=None)

# 1列目を「output_konck12_col1.txt」に，2列目を「output_konck12_col2.txt」に保存
data[0].to_csv('output_knock12_col1.txt', index=False, header=None)
data[1].to_csv('output_knock12_col2.txt', index=False, header=None)

# 確認
# cutコマンドで文字列を切り出す
