# 94. ビーム探索
# 91で学習したニューラル機械翻訳モデルで翻訳文をデコードする際に，ビーム探索を導入せよ．
# ビーム幅を1から100くらいまで適当に変化させながら，開発セット上のBLEUスコアの変化をプロットせよ．

'''
#!/usr/bin/zsh

set -e
set -x

if [ $# -ge 4 ]; then
    input_file=$1
    output_dir=$2
    device=$3
    model_path=$4
else
    echo "Please specify the paths to the input_file and output directory"
    echo "Usage: `basename $0` <input_file> <output_dir> <gpu-device-num(e.g: 0)> <path to model_file/dir> [optional args: <path-to-reranker-weights> <featuers,e.g:eo,eolm]"   >&2
fi
if [[ -d "$model_path" ]]; then
    models=`ls $model_path/*pt | tr '\n' ' ' | sed "s| \([^$]\)| --path \1|g"`
    echo $models
elif [[ -f "$model_path" ]]; then
    models=$model_path
elif [[ ! -e "$model_path" ]]; then
    echo "Model path not found: $model_path"
fi


for i in `seq 100`
do
beam=$i
nbest=$beam

# running fairseq on the test data
CUDA_VISIBLE_DEVICES=$device fairseq-generate /work/aomi/100knock2020/chapter10/data/processed_16000/bin --gen-subset valid --path $models --beam $beam --nbest $beam --remove-bpe --batch-size 32 > output.nbest.txt

python sort_output.py output.nbest.txt $beam

perl /work/aomi/100knock2020/chapter10/tools/multi-bleu.perl /work/aomi/100knock2020/chapter10/data/dev.tok.en < output.sorted.txt >> bleu.txt
done
'''