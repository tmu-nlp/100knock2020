# 92. 機械翻訳モデルの適用
# 91で学習したニューラル機械翻訳モデルを用い，
# 与えられた（任意の）日本語の文を英語に翻訳するプログラムを実装せよ．


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

beam=1
nbest=$beam

# running fairseq on the test data
CUDA_VISIBLE_DEVICES=$device fairseq-interactive /work/aomi/100knock2020/chapter10/data/bin --path $models --beam $beam --nbest $beam < $input_file > output.nbest.txt

# getting best hypotheses
cat output.nbest.txt | grep "^H"  | python -c "import sys; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if(i%$nbest == 0) ]); print(x)" | cut -f3 > output.txt
'''