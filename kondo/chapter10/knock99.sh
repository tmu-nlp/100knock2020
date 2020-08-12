set -e
set -x
script_dir=scripts
output_dir=output/model_1m
mkdir -p $output_dir
SRC_PATH=use_file/tok/kyoto-test.ja
#bpe_model=model/bpe.model
#subword-nmt apply-bpe -c $bpe_model < $test_file > $output_dir/test.bpe.txt
CUDA_VISIBLE_DEVICES=$1 PYTHONIOENCODING=utf-8 fairseq-interactive \
    ./use_file/AT_bin \
    --path model/model1_m/checkpoint10.pt \
    --source-lang ja --target-lang en\
    --beam 1 < $SRC_PATH > $output_dir/output.bpe.nbest.txt
# getting best hypotheses
cat $output_dir/output.bpe.nbest.txt | grep "^H"  | python ${script_dir}/sort.py 1 $output_dir/output.bpe.txt
# debpe
cat $output_dir/output.bpe.txt | sed 's|@@ ||g' > $output_dir/output.tok.txt