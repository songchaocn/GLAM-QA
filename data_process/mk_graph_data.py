
import torch

from torch_geometric.data import Data

from tqdm import tqdm

import json

import pandas as pd
import torch.nn.functional as F


import sys
import os

current_file_path = os.path.abspath(__file__)
glam_qa_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, glam_qa_dir)
# sys.path.append('/mnt/cxy/cv_MRG')

# ========= 所有自建模块均显式从 my_code 导入 =========
from my_code import sampler, layers, my_config, conversation, language_model, graph_transformer
from my_code.my_config import *                                    # 原 from my_config import *
from my_code.language_model import InstructGLM2
from my_code.conversation import conv_templates, SeparatorStyle

# ========= Hugging Face  transformers =========
from transformers import (
    AutoTokenizer,
)

device = torch.device('cuda:0')


def process_csv_and_create_tensors(file_path):
    
    df = pd.read_csv(file_path,sep=';')
    edge_index = [df["src"].tolist(), df["dst"].tolist()]
    return edge_index

os.environ["TOKENIZERS_PARALLELISM"] = "false"

IGNORE_TOKEN_ID = -100




def preprocess(prompt, gpt, tokenizer, max_length, mode='train'):
    conv = conv_templates["qwen2.5_7b"].copy()
    assert conv.sep_style == SeparatorStyle.TWO

    roles = conv.roles
    tokenizer.padding_side = 'right' if mode == 'train' else 'left'

    # Apply prompt templates
    conversations = []
    conv.append_message(roles[0], prompt)
    if mode == 'train':
        conv.append_message(roles[1], gpt)
    else:
        conv.append_message(roles[1], None)
    conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    # if mode != 'train':
    if mode == 'test':
        end_token = "<|endoftext|>"
        targets = tokenizer(
            [gpt+end_token],
            return_tensors="pt",
            padding="max_length",
            max_length=200,
            truncation=True,
        ).input_ids
    else:
        targets = input_ids.clone()

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())


            turns = conversation.split(conv.sep2)
  
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)

                if len(parts) != 2:
                    break
                parts[0] += sep
                # # "-2" is hardcoded for the Llama tokenizer to make the offset correct. the first label is not _, but _label
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                # Remove the -2 operation

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

            target[cur_len:] = IGNORE_TOKEN_ID


            if cur_len < max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)\n"
                        # f"conversations:{conversations[0]}"
                    )

    return dict(
        input_ids=input_ids,
        target_ids=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        text=conversations[0]
    )


args = parse_args()
llm_device = torch.device('cpu')
device = torch.device('cuda:1')

model_path = 'your_path/llm/Qwen2.5-7B'


llm = InstructGLM2.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(llm_device)

llama_embeds = llm.get_input_embeddings().weight
print('llama_embeds.shape:',llama_embeds.shape)

tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.add_special_tokens({'unk_token': '<unk>'})

tokenizer.pad_token = tokenizer.unk_token

tokenizer.add_special_tokens({'additional_special_tokens': ['<t_patch>']})


# llm.resize_token_embeddings(len(tokenizer), mean_resizing=False)
llm.resize_token_embeddings(len(tokenizer))  
llama_embeds = llm.get_input_embeddings().weight
print('llama_embeds.shape:',llama_embeds.shape)


def mk_data(stage_name,data_cls):
    
    instructions_path = f'../data/{data_cls}/qa_data/q2tri_{stage_name}.json'

    with open(instructions_path, 'r', encoding='utf-8') as file:
        instructions = json.load(file)
    

    print("len(instructions):",len(instructions))
    cnt = 0
    max_text_length = 400
    for idx in tqdm(range(len(instructions)), desc="Processing", unit="instruction"):
        
        instruction = instructions[idx]
        tokens = " ".join([f"<t_patch>" for i in range (1, 1 + args.num_token)])
        question = instruction['question']

        gpt = instruction['answer']
    
        prompt = f'Given a representation of triples: <Node 1>, Please answer the following question based on the information of the triples,and provide a detailed explanation.\n' + f'Question: {question}\n'
        prompt = prompt.replace("<Node 1>", tokens)

        out_dict = preprocess(prompt,gpt, tokenizer, max_text_length, mode=stage_name)
        
        
        graph = Data()

        graph.x = torch.load(f'../data/{data_cls}/emb/{stage_name}/{idx}.pt').to(dtype=torch.bfloat16)
        
        graph.edge_index = torch.LongTensor(process_csv_and_create_tensors(f'../data/{data_cls}/edge_index/{stage_name}/{idx}.csv'))
        
        graph.edge_attr = None

        node_count = graph.x.size(0)
        is_node = (out_dict['input_ids'] >= 151665)#qwen2.5
        

        input_ids = out_dict['input_ids']
        answer_ids = out_dict['target_ids']
        attention_mask = out_dict['attention_mask']

        
        graph.is_node = is_node
        graph.input_ids = input_ids
        graph.target_ids = answer_ids
        graph.attention_mask = attention_mask
        graph.num_nodes = node_count
        graph.graphID = cnt
        
        graph.to(device, non_blocking=True)

        save_folder = f'../Data/ultra/{data_cls}/{stage_name}_qa_qwen2.5_7b'
        
        os.makedirs(save_folder, exist_ok=True)
        save_path = save_folder + '/' + f'{cnt}.pt'
        
        torch.save(graph, save_path)
        cnt += 1


    print(f"{stage_name} cnt:",cnt)
    try:
        embeddings = torch.load(save_path)
        print('embeddings shape', embeddings.x.shape)  # 
        print(embeddings)
        
    except FileNotFoundError:
        print(f"未找到文件: {save_path}")

    print(f'{stage_name} cnt:{cnt}')
    
        
stage_name_list = ['train','test']

data_cls = 'PubMedQA'####MedQA-en,MedMCQA,PubMedQA
for stage_name in stage_name_list:
    print(f'{stage_name} set:')
    mk_data(stage_name,data_cls)


import torch
ggg = torch.load(f'../Data/ultra/{data_cls}/{stage_name}_qa_qwen2.5_7b/0.pt')

print(ggg)
