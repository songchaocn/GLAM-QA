



from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm
import pandas as pd

import os

import unittest
import string
import re
import collections
import time

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1





from bert_score import BERTScorer

#  BERTScorer
scorer = BERTScorer(model_type='bert-base-uncased', lang='en', rescale_with_baseline=False)


def bert_similarity(candidate, reference):

    cand_list = [truncate_text(candidate)] if isinstance(candidate, str) else [truncate_text(c) for c in candidate]
    ref_list = [truncate_text(reference)] if isinstance(reference, str) else [truncate_text(r) for r in reference]
    

    P, R, F1 = scorer.score(cand_list, ref_list)
    return P.item(), R.item(), F1.item()



###########################################################################################

model_name = "path_to_your_models/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_cls = 'PubMedQA'####MedQA-en,MedMCQA,PubMedQA
date_flag = time.strftime("%m%d", time.localtime())

def truncate_text(text, max_length=512):
    """截断文本到最大长度"""
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:  #
        tokens = tokens[:max_length - 2]
    return tokenizer.convert_tokens_to_string(tokens)

with open(f'../data/{data_cls}/qa_data/q2tri_test.json', 'r', encoding='utf-8') as file:
    qa_gold = json.load(file)  


with open(f'../saved_output/{data_cls}/output_json/test_pred_{date_flag}.json', 'r', encoding='utf-8') as file:
    qa_result = json.load(file)  



id_list = []
bert_score_list1 = []
P_bert_score_list1 = []
R_bert_score_list1 = []
f1_score_list1 = []

for idx in tqdm(range(len(qa_gold)), desc="Processing", unit=""):

    gold_a = qa_gold[idx]['answer']
    
    generated_text = qa_result[idx]['generated_text']

    
    if not generated_text or not generated_text.strip():
        #P, R, F1 
        bert_score1 = 0
        R_bert_score1 = 0
        P_bert_score1 = 0
        f1_score1 = 0
        print(f"Warning: Empty generated_text for question ID {idx}")
    else:
        P_bert_score1,R_bert_score1,bert_score1 = bert_similarity(generated_text, gold_a)
        f1_score1 = compute_f1(generated_text, gold_a)


    
    id_list.append(idx)
    bert_score_list1.append(bert_score1)
    P_bert_score_list1.append(P_bert_score1)
    R_bert_score_list1.append(R_bert_score1)
    
    f1_score_list1.append(f1_score1)

tmp_dict_bert1 = {
    'id_list':id_list,
    'P':P_bert_score_list1,
    'R':R_bert_score_list1,
    'score_list':bert_score_list1,
    
}

tmp_dict_f1_1 = {
    'id_list':id_list,
    'score_list':f1_score_list1,
}



df_bert_1 = pd.DataFrame(tmp_dict_bert1)
df_f1_1 = pd.DataFrame(tmp_dict_f1_1)

save_folder = f'./{data_cls}/output_csv'
os.makedirs(save_folder, exist_ok=True)
save_bert_path = os.path.join(save_folder, f'bert_score_f1_{date_flag}.csv')
save_f1_path = os.path.join(save_folder, f'f1_score_{date_flag}.csv')
df_bert_1.to_csv(save_bert_path, index=False, encoding='utf-8')
df_f1_1.to_csv(save_f1_path, index=False, encoding='utf-8')