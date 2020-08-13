# 91. 機械翻訳モデルの訓練
# 90で準備したデータを用いて，ニューラル機械翻訳のモデルを学習せよ
# （ニューラルネットワークのモデルはTransformerやLSTMなど適当に選んでよい）．


'''
CUDA_VISIBLE_DEVICES=6 nohup fairseq-train \
  /work/aomi/100knock2020/chapter10/data/processed/bin \
  --save-dir /work/aomi/100knock2020/chapter10/knock91/models/model_1111 \
  --arch transformer \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr 0.0005 --lr-scheduler inverse_sqrt \
  --min-lr '1e-09' --warmup-init-lr '1e-07' \
  --warmup-updates 4000 \
  --dropout 0.3 \
  --max-epoch 40 \
  --clip-norm 1.0 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 1024 \
  --seed 1111 > train.log &
'''
