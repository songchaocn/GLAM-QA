#####################################################get_triples new

import os
import json
import csv
import json
import pandas as pd
from tqdm import tqdm
import os
import torch
##1.get_triples new

def get_csv_num(path):
        # 定义目录路径
        directory_path = path

        # 获取目录下的所有文件和子目录名
        files_and_dirs = os.listdir(directory_path)

        # 计算CSV文件的数量
        csv_files_count = sum(file.endswith('.csv') for file in files_and_dirs if os.path.isfile(os.path.join(directory_path, file)))
        return csv_files_count

def get_t_graph(in_path,out_path):
        data = []
        
        with open(in_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=';')
            for row in reader:
                data.append(row)

        connections = []
        num_triples = len(data)
        for i in range(num_triples):
            triple_i = [data[i]['head'], data[i]['tail']]#
            for j in range(i + 1, num_triples):
                triple_j = [data[j]['head'], data[j]['tail']]#
                if any(entity in triple_j for entity in triple_i):
                    connections.append((i, j))

        # 将连接关系写入新的CSV文件
        with open(out_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['src', 'dst'])
            for src, dist in connections:
                writer.writerow([src, dist])
        
        

data_cls = 'PubMedQA'##MedQA-en,MedMCQA,PubMedQA

stage_list = ['train','test']
for flag_tail in stage_list:
    print(f'正在处理 {flag_tail} 集合...')
    # 1. 路径
    json_file   = f'../data/{data_cls}/qa_data/q2tri_{flag_tail}.json'  
    out_dir     = f'../data/{data_cls}/triples/{flag_tail}'   # 保存 0.csv 1.csv ... 的文件夹
    os.makedirs(out_dir, exist_ok=True)


    # 3. 加载 json
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for i in range(len(data)):
        question = data[i]['question']
        triples_list = data[i]['triples']
        unique_triples_list = [list(t) for t in {tuple(row) for row in triples_list}]

        csv_file = os.path.join(out_dir, f'{i}.csv')
        try:
            with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(['head', 'relation', 'tail'])
                for j in range(len(unique_triples_list)):
                    head = unique_triples_list[j][0]
                    relation = unique_triples_list[j][1]
                    tail = unique_triples_list[j][2]
                    writer.writerow([head, relation, tail])
                        
        except Exception as e:
            print(f"写入文件时出现错误: {e},{csv_file}")  
    print('全部写完，保存在', out_dir)
    # # ##########get all triples
    with open(f'../data/{data_cls}/qa_data/q2tri.json', 'r', encoding='utf-8') as file:
        all_data = json.load(file)

    total_list = []
    for i in range(len(all_data)):
        question = all_data[i]['question']
        triples_list = all_data[i]['triples']
        # unique_triples_list = [list(t) for t in {tuple(row) for row in triples_list}]
        total_list.extend(triples_list)
        
    unique_triples_list = [list(t) for t in {tuple(row) for row in total_list}]

    csv_file = f'../data/{data_cls}/triples/triples.csv'
    try:
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            # 写入表头
            writer.writerow(['head', 'relation', 'tail'])
            for j in range(len(unique_triples_list)):
                head = unique_triples_list[j][0]
                relation = unique_triples_list[j][1]
                tail = unique_triples_list[j][2]
                writer.writerow([head, relation, tail])
                        
    except Exception as e:
        print(f"写入文件时出现错误: {e},{csv_file}")  
    print('全部写完，保存在', csv_file)
    
    
    ####3.get edge index 
    
    directory_path = f'../data/{data_cls}/triples/{flag_tail}'
    csv_files_count = get_csv_num(directory_path)


    for i in range(csv_files_count):

        in_path = f'../data/{data_cls}/triples/{flag_tail}/{i}.csv'

        out_folder = f'../data/{data_cls}/edge_index/{flag_tail}'
        os.makedirs(out_folder, exist_ok=True)
        out_path = out_folder + '/' + f'{i}.csv'
        
        get_t_graph(in_path,out_path)
    
    
    # #2.get sub emb
    # total_emb_path = f'../images_data/{data_cls}/medical_triples_emb.pt'#初始化还是用nomic
    # out_dir     = f'../images_data/{data_cls}/clip_data/kg_emb/{flag_tail}'          # 保存 0.pt 1.pt ...
    # os.makedirs(out_dir, exist_ok=True)

    # # 2. 加载全部 embedding（可放 GPU，也可 CPU）
    # total_embeddings = torch.load(total_emb_path)          # shape: (N, dim)
    # cnt2 = 0
    # # 3. 加载 json
    # with open(json_file, 'r', encoding='utf-8') as f:
    #     data = json.load(f)

    # # 4. 逐 image 写出
    # for item in tqdm(data, desc='写 pt'):
    #     # img_id  = item['id']
    #     img_id  = cnt2
    #     kg_id   = item['kg_ids']               # list[int]
    #     if len(kg_id) == 0:
    #         # print(img_id)
    #         continue
    #     kg_emb  = total_embeddings[kg_id]     # (k, dim)

    #     out_path = os.path.join(out_dir, f'{img_id}.pt')
    #     torch.save(kg_emb, out_path)
    #     cnt2 += 1
    # print(kg_emb.shape)
    # print('全部写完 ->', out_dir)


