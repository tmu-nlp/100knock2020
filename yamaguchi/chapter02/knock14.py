# コマンドライン引数のためのモジュールと，データ処理ライブラリ「Pandas」をインポート
import sys
import pandas as pd

# リストの0番目がファイル名，1番目がコマンドライン引数となる．
N = int(sys.argv[1])

# 区切り文字がタブ(sep='\t')なので「read_table()」を用いる
# headerがないので「None」を指定
data = pd.read_table('popular-names.txt', sep='\t', header=None)

# 結果を表示
# 「head」で上からの行数を指定
print(data.head(N))

# 実行時は「python knock14.py 3」などとして実行

# 確認
# headコマンドで最初の指定行分だけ表示させる
