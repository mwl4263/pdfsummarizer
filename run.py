from transformers import pipeline
from PyPDF2 import PdfReader
from tkinter import *   # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
from tqdm import tqdm
#set up file picker
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(defaultextension=".pdf") # show an "Open" dialog box and return the path to the selected file
print("=== Please wait while I parse the PDF... ===")

summarizer = pipeline("summarization", model = "sshleifer/distilbart-cnn-12-6")

model = T5ForConditionalGeneration.from_pretrained("Unbabel/gec-t5_small")
tokenizer = T5Tokenizer.from_pretrained('t5-small')
reader = PdfReader(filename)
text = ""

with open ("output.md", "w") as f:
    f.write("# Summary\n\n")
    f.write("#### Generated using AI language models: distilbart and gec-t5_small for grammar correction\n\n")
    print("=== SUMMARY GENERATION HAS BEGUN ===")
    print("=== Pages Summarized: ===")
    for x in tqdm(range(len(reader.pages))):
        text = reader.pages[x].extract_text()
        response = summarizer(text)
        text_response= response[0]['summary_text']
        tokenized_sentence = tokenizer('gec: ' + text_response, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        corrected_sentence = tokenizer.decode(
        model.generate(
            input_ids = tokenized_sentence.input_ids,
            attention_mask = tokenized_sentence.attention_mask,
            max_length=128,
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
print("=== The summary has been generated.  Open summary.pdf to view the summary. ===")
os.system("md2pdf output.md summary.pdf")