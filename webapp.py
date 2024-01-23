from flask import Flask, render_template
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from transformers import pipeline
from PyPDF2 import PdfReader
from tkinter import *   # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename, asksaveasfile
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
app = Flask(__name__)


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
@app.route('/upload_success')
def done():
    print("done")
    return render_template('upload.html')


@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    filename = request.args.get('filename')
    summarizer = pipeline("summarization", model = "facebook/bart-large-cnn")

    model = T5ForConditionalGeneration.from_pretrained("Unbabel/gec-t5_small")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    reader = PdfReader("uploads/" + filename)
    text = ""
    with open ("static/summary.txt", "w") as f:
        print("=== END IGNORE ===")
        f.write("# Summary\n\n")
        f.write("#### Generated using AI language models: bart-large-cnn and gec-t5_small for grammar correction\n\n")
        print("=== Summary generation has begun. ===")
        print("=== Pages Summarized: ===")
        for x in tqdm(range(len(reader.pages))):
            text = reader.pages[x].extract_text()
            response = summarizer(text)
            text_response= response[0]['summary_text']
            tokenized_sentence = tokenizer('gec: ' + text_response, max_length=300, truncation=True, padding='max_length', return_tensors='pt')
            corrected_sentence = tokenizer.decode(
            model.generate(
                input_ids = tokenized_sentence.input_ids,
                attention_mask = tokenized_sentence.attention_mask,
                max_length=300,
                num_beams=5,
                early_stopping=True,
                    )[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            if len(corrected_sentence) < 1:
                continue
            else:
                f.write("- " + corrected_sentence + "\n\n")

        f.close()
    print("- done -")
    os.system("rm uploads/" + filename)
    return redirect(url_for('done'))

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print("upload_file")
    print(request.files)
    if request.method == 'POST':
        print("request.method == 'POST'")
        # check if the post request has the file part
        if 'fileupload' not in request.files:
            print("file not in request.files")
            flash('No file part')
            return redirect(request.url)
        file = request.files['fileupload']
        print("file: ", file)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print("file.filename == ''")
            flash('No selected file')
            return redirect(request.url)
        if file:
            print("if file")
            filename = secure_filename(file.filename)
            print("filename: ", filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('summarize',filename=filename))
    return render_template('index.html')



if __name__ == '__main__':
    app.secret_key = "SDJFAKDSGJNLAND#!@#$2134325"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run(debug=True)
