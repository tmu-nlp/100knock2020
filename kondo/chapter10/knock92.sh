"""
92. 機械翻訳モデルの適用Permalink
91で学習したニューラル機械翻訳モデルを用い，与えられた（任意の）日本語の文を英語に翻訳するプログラムを実装せよ．
"""

set -e
set -x
script_dir=scripts
output_dir=output/model_1m
mkdir -p $output_dir
#bpe_model=model/bpe.model
#subword-nmt apply-bpe -c $bpe_model < $test_file > $output_dir/test.bpe.txt
nbest=1
CUDA_VISIBLE_DEVICES=$1 PYTHONIOENCODING=utf-8 fairseq-generate \
    ./use_file/AT_bin \
    --path model/model1_m/checkpoint10.pt \
    --iter-decode-max-iter 9 \
    --beam 1 \
    --batch-size 128 > $output_dir/output.bpe.nbest.txt
# getting best hypotheses
cat $output_dir/output.bpe.nbest.txt | grep "^H"  | python ${script_dir}/sort.py 1 $output_dir/output.bpe.txt
# debpe
cat $output_dir/output.bpe.txt | sed 's|@@ ||g' > $output_dir/output.tok.txt