set -e
set -x
script_dir=scripts
output_dir=output/bpe
mkdir -p $output_dir
#bpe_model=model/bpe.model
#subword-nmt apply-bpe -c $bpe_model < $test_file > $output_dir/test.bpe.txt
nbest=1
CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 fairseq-generate \
    ./use_file_bpe/AT_bin \
    --path model/bpe/checkpoint10.pt \
    --iter-decode-max-iter 9 \
    --beam 1 \
    --batch-size 128 > $output_dir/output.bpe.nbest.txt
# getting best hypotheses
cat $output_dir/output.bpe.nbest.txt | grep "^H"  | python ${script_dir}/sort.py 1 $output_dir/output.bpe.txt
# debpe
cat $output_dir/output.bpe.txt | sed 's|@@ ||g' > $output_dir/output.tok.txt