from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import os
import numpy as np
from translation import *
import sentencepiece as spm
import re
from fairseq.models.transformer import TransformerModel

app = Flask(__name__)

######## Preparing the translator
#cur_dir = os.path.dirname(__file__)
sp = spm.SentencePieceProcessor()
sp.load("models/jsec.ja.model")

ja2en = TransformerModel.from_pretrained(
    'checkpoints/98subwords/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='data/bin/98_subwords/'
)

######## Flask
class TextForm(Form):
    source = TextAreaField('', [validators.DataRequired(), validators.length(min=5)])

@app.route('/')
def index():
    form = TextForm(request.form)
    #text = translate(form)
    return render_template('textform.html', form=form, target=None)

@app.route('/', methods=['POST'])
def results():
    form = TextForm(request.form)
    if request.method == 'POST' and form.validate():
        source = request.form['source']
        target = translate(source)
        
        return render_template('textform.html',
                                #source=source,
                                form=form,
                                target=target)
        
    #return render_template('reviewform.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)