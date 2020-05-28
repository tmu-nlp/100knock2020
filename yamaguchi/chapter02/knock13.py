# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

# knock12で保存したテキストを読み込む
# headerがないので「None」を指定
data1 = pd.read_csv('output_knock12_col1.txt', header=None)
data2 = pd.read_csv('output_knock12_col2.txt', header=None)

# 「concat()」関数で結合する
# 連結方向を「axis=1」として指定．「0」だと縦方向に連結される．
connection = pd.concat([data1, data2], axis=1)

# 区切り文字をタブ(sep='\t')として保存
# indexもheaderも不要
connection.to_csv('ouotput_knock13.txt', sep='\t', index=False, header=None)

# 確認
# sedコマンドで文字列の連結を行う
