# 93. BLEUスコアの計測
# 91で学習したニューラル機械翻訳モデルの品質を調べるため，
# 評価データにおけるBLEUスコアを測定せよ．


# mosesのプログラムを使用した
'''
https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
'''

# BLEUを測る
'''
perl multi-bleu.perl kyoto-test.en < output.tok.txt
'''

# 結果
'''
BLEU = 6.05, 33.5/8.5/3.5/1.8 (BP=0.931, ratio=0.933, hyp_len=24938, ref_len=26734)
'''