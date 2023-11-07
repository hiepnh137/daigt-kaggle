import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

model_name = 'tuner007/pegasus_paraphrase'
# model_name = 'prithivida/parrot_paraphraser_on_T5'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams):
    batch = tokenizer([input_text],truncation=True,padding='longest',max_length=100, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=100,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

data = pd.read_csv('data/train_essays_1.0.csv')
data = data.loc[data['generated']==1]
corpus = data['text']

paraphrased_corpus = []
for doc in tqdm(corpus):
    paraphrased_sentences = []
    paragraphs = doc.split('\n')
    for para in paragraphs:
        paraphrased_para = []
        sentences = sent_tokenize(para)
        for sent in sentences:
            paraphrased_para.append(get_response(sent, num_return_sequences=1, num_beams=3)[0])
        paraphrased_sentences.extend(paraphrased_para)
        paraphrased_sentences.append('\n')
    paraphrased_corpus.append(' '.join(paraphrased_sentences[:-1]))

data['text'] = paraphrased_corpus
data.to_csv('data/paraphrased_essays_1.0.csv')