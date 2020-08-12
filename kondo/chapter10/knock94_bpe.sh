set -e
set -x
script_dir=scripts
output_dir=output/model_1m
mkdir -p $output_dir
dir=/work/seiichiro/100knock2020/chapter10
#bpe_model=model/bpe.model
#subword-nmt apply-bpe -c $bpe_model < $test_file > $output_dir/test.bpe.txt
nbest=1
for i in `seq 1 100`
do
CUDA_VISIBLE_DEVICES=$1 PYTHONIOENCODING=utf-8 fairseq-generate \
    ./use_file_bpe/AT_bin \
    --path model/bpe/checkpoint10.pt \
    --iter-decode-max-iter 9 \
    --beam $i \
    --batch-size 128 > $output_dir/output.bpe.nbest.txt
# getting best hypotheses
cat $output_dir/output.bpe.nbest.txt | grep "^H"  | python ${script_dir}/sort.py 1 $output_dir/output.bpe.txt
# debpe
cat $output_dir/output.bpe.txt | sed 's|@@ ||g' > $output_dir/output.tok.txt

sacrebleu $dir/use_file/tok/kyoto-test.en <$dir/output/model_1m/output.tok.txt | awk -F" " '{print $3}' >> score_bpe.text
done

python plot.py score_bpe.text