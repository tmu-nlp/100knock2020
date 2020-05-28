# コマンドライン引数のためのモジュールと，データ処理ライブラリ「Pandas」をインポート
# 計算のための「math」もインポート
import sys
import pandas as pd
import math

# リストの0番目がファイル名，1番目がコマンドライン引数となる．
N = int(sys.argv[1])

# 区切り文字がタブ(sep='\t')なので「read_table()」を用いる
# headerがないので「None」を指定
data = pd.read_table('popular-names.txt', sep='\t', header=None)

# 各ファイルの行数を切り上げで確認する
number_of_lines = math.ceil(len(data) / N)

# 分割した行ごとにファイルを保存する
# 「loc[]」で行数を指定
# 区切り文字をタブ(sep='\t')として保存
# indexもheaderも不要
for i in range(N):
  data.loc[number_of_lines * i : number_of_lines * (i + 1)].to_csv(f'output_knock16_{i + 1}.txt', sep='\t', index=False, header=None)

# 実行時は「python knock16.py 3」などとして実行

# 確認
# splitコマンドでファイルを分割する
