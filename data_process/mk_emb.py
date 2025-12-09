from langchain_ollama import OllamaEmbeddings
import pandas as pd
import torch

from tqdm import tqdm
import time
import os
import csv



def get_csv_num(path):
    # 定义目录路径
    directory_path = path

    # 获取目录下的所有文件和子目录名
    files_and_dirs = os.listdir(directory_path)

    # 计算CSV文件的数量
    csv_files_count = sum(file.endswith('.csv') for file in files_and_dirs if os.path.isfile(os.path.join(directory_path, file)))
    return csv_files_count


def get_nodes_embedding(data1,wrt_path):
    qa_tensors_list = []  # 使用列表来收集输出张量
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    white_color = '\033[37m'
    reset_color = '\033[0m'

    # 自定义进度条格式，使用白色
    bar_format = "{l_bar}%s{bar}%s{r_bar}" % (white_color, reset_color)
    for i in tqdm(range(len(data1)), desc="get node", bar_format=bar_format): 
        head = str(data1.loc[i]['head'])
        relation = str(data1.loc[i]['relation'])
        tail = str(data1.loc[i]['tail'])
        question = str(head.replace('_',' ')) + ' ' + str(relation.replace('_',' ')) + ' ' + str(tail.replace('_',' '))
        try:
            e = embeddings.embed_query(str(question))
            qa_tensors_list.append(e)
        except:
           print(question)

    output_tensor = torch.tensor(qa_tensors_list).to(device)
    
    save_path =wrt_path
    torch.save(output_tensor, save_path)

    
    print("output_tensor.shape:",output_tensor.shape)


start_time = time.time()




device = torch.device('cuda:0')

data_cls = 'PubMedQA'##MedQA-en,MedMCQA,PubMedQA
data_csv = pd.read_csv(f'../data/{data_cls}/triples/triples.csv',sep=';')



wrt_path = f'../data/{data_cls}/triples/triples_emb.pt'

get_nodes_embedding(data_csv,wrt_path)
print(f'triples_emb saved {wrt_path}')

stage_list = ['train','test']

for flag_tail in stage_list:
    print(f'正在处理 {flag_tail} 集合...')

    directory_path = f'../data/{data_cls}/triples/{flag_tail}'
    csv_files_count = get_csv_num(directory_path)



    total_embeddings = torch.load(wrt_path).to(device)
    print(total_embeddings.shape)
    triple_dict = {}
    with open(f'../data/{data_cls}/triples/triples.csv', 'r', encoding='utf-8') as f_total:
        reader_total = csv.DictReader(f_total, delimiter=';')
        for idx, row in enumerate(reader_total, start=1):  # 索引从1开始（跳过表头行）
            head = row['head'].strip()
            relation = row['relation'].strip()
            tail = row['tail'].strip()
            triple = (head, relation, tail)
            if triple not in triple_dict:  # 避免重复三元组覆盖索引
                triple_dict[triple] = idx

    # 检查0.csv中的三元组
    for i in range(csv_files_count):

        in_path = f'../data/{data_cls}/triples/{flag_tail}/{i}.csv'
        result_indices = []
        missing_triples = []
        with open(in_path, 'r', encoding='utf-8') as f_test:
            reader_test = csv.DictReader(f_test, delimiter=';')
            for row in reader_test:
                head = row['head'].strip()
                relation = row['relation'].strip()
                tail = row['tail'].strip()
                current_triple = (head, relation, tail)
                if current_triple in triple_dict:
                    result_indices.append(triple_dict[current_triple])
                else:
                    missing_triples.append(current_triple)
                
                
            adjusted_indices = [idx - 1 for idx in result_indices]

            indices_tensor = torch.tensor(adjusted_indices, dtype=torch.long).to(device)

            test_embeddings = total_embeddings.index_select(dim=0, index=indices_tensor).to(device)
            out_folder = f'../data/{data_cls}/emb/{flag_tail}'
            os.makedirs(out_folder, exist_ok=True)
            
            save_path =out_folder + f"/{i}.pt"
            torch.save(test_embeddings, save_path)
    
    print("提取的嵌入形状：", test_embeddings.shape)

end_time = time.time()
elapsed_time = end_time - start_time  # 计算时间差，得到所用时间
print(f"所用时间: {elapsed_time} 秒")