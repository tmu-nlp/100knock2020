from flask_ngrok import run_with_ngrok
from flask import Flask, request, render_template
from wtforms import Form, TextAreaField, validators
import subprocess
import MeCab

app = Flask(__name__, template_folder='templates')
run_with_ngrok(app)   #starts ngrok when the app is run

tagger = MeCab.Tagger("-Owakati")
cmd = 'fairseq-interactive data-bin/kftt.ja-en --path checkpoints/kftt.ja-en/checkpoint_best.pt'

class InputForm(Form):
  inputsent = TextAreaField('', [validators.DataRequired(), validators.length(min=15)])

@app.route("/")
def home():
  form = InputForm(request.form)
  return render_template('form.html', form=form)

@app.route('/results', methods=['POST'])
def results():
  form = InputForm(request.form)
  if request.method == 'POST' and form.validate():
    input_sent = request.form['inputsent']
    input_sent = tagger.parse(input_sent)
    with open('input.txt', 'w') as fp:
      fp.write(f'{input_sent}\n')
    proc = subprocess.run(['fairseq-interactive', 'data-bin/kftt.ja-en', '--path', 'checkpoints/kftt.ja-en/checkpoint_best.pt', '--input', 'input.txt'], encoding='UTF-8', stdout=subprocess.PIPE)
    output_sent = proc.stdout.strip().split('\n')[8].split('\t')[-1]
    return render_template('results.html', input_sent=input_sent, output_sent=output_sent)
  return render_template('form.html', form=form)

if __name__ == '__main__':
  app.run()

