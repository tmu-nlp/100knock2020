# データ処理ライブラリ「Pandas」をインポート
import pandas as pd

# 区切り文字がタブ(sep='\t')なので「read_table()」を用いる
# headerがないので「None」を指定
data = pd.read_table('popular-names.txt', sep='\t', header=None)

# 「data[0]」で1列目を指定
# 「value_counts()」で出現頻度を数え，出現頻度の高い順に並べ替える．
print(data[0].value_counts())

# 確認
# cutコマンドで1列目を指定し，uniqコマンドで出現頻度を数えて重複を削除し，sortコマンドで並べ替える．
