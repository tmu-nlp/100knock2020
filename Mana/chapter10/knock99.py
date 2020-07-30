from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from fairseq.models.transformer import TransformerModel


# pytorch fairseqができるならこれで動く気がするが...
# うまくできるか？

app = Flask(__name__)

class TransForm(Form):
    entersent = TextAreaField('',[validators.DataRequired()])

@app.route('/')
def index():
    form = TransForm(request.form)
    return render_template('app.html', form=form)

@app.route('/trans', methods=['POST'])
def trans():
    ja2en = TransformerModel.from_pretrained('/path/to/checkpoints',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data-bin/wmt17_zh_en_full',
    bpe='subword_nmt',
    bpe_codes='data-bin/wmt17_zh_en_full/zh.code'
    )
    form = TransForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['entersent']
        name = ja2en.translate(name)
        return render_template('translated.html', name=name)
    return render_template('app.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5800, threaded=True)