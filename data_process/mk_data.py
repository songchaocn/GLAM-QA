
import os
import json
import csv
import json
import pandas as pd
from tqdm import tqdm
import os
import torch


def get_csv_num(path):

        directory_path = path
        files_and_dirs = os.listdir(directory_path)
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

        with open(out_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['src', 'dst'])
            for src, dist in connections:
                writer.writerow([src, dist])
        
        

data_cls = 'PubMedQA'##MedQA-en,MedMCQA,PubMedQA

stage_list = ['train','test']
for flag_tail in stage_list:
    print(f'Processing {flag_tail} set...')

    json_file   = f'../data/{data_cls}/qa_data/q2tri_{flag_tail}.json'  
    out_dir     = f'../data/{data_cls}/triples/{flag_tail}'   
    os.makedirs(out_dir, exist_ok=True)



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
            print(f"Error writing file: {e},{csv_file}")  
    print('All files written and saved to', out_dir)
    with open(f'../data/{data_cls}/qa_data/q2tri.json', 'r', encoding='utf-8') as file:
        all_data = json.load(file)

    total_list = []
    for i in range(len(all_data)):
        question = all_data[i]['question']
        triples_list = all_data[i]['triples']
        total_list.extend(triples_list)
        
    unique_triples_list = [list(t) for t in {tuple(row) for row in total_list}]

    csv_file = f'../data/{data_cls}/triples/triples.csv'
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
        print(f"Error writing file: {e},{csv_file}")  
    print('All files written and saved to', csv_file)
    

    directory_path = f'../data/{data_cls}/triples/{flag_tail}'
    csv_files_count = get_csv_num(directory_path)


    for i in range(csv_files_count):

        in_path = f'../data/{data_cls}/triples/{flag_tail}/{i}.csv'

        out_folder = f'../data/{data_cls}/edge_index/{flag_tail}'
        os.makedirs(out_folder, exist_ok=True)
        out_path = out_folder + '/' + f'{i}.csv'
        
        get_t_graph(in_path,out_path)
    
    


