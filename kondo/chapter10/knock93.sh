"""
93. BLEUスコアの計測Permalink
91で学習したニューラル機械翻訳モデルの品質を調べるため，評価データにおけるBLEUスコアを測定せよ．
"""


dir=/work/seiichiro/100knock2020/chapter10

sacrebleu $dir/use_file/kyoto-test.en <$dir/output/model_1m/output.tok.txt