# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

# 区切り文字がタブ(sep='\t')なので「read_table()」を用いる
# headerがないので「None」を指定
data = pd.read_table('popular-names.txt', sep='\t', header=None)

# 半角スペースを区切り文字として保存
# indexもheaderも不要
data.to_csv('output_knock11.txt', sep=' ', index=False, header=None)

# 確認
# sedコマンドで文字列の置換を行う
