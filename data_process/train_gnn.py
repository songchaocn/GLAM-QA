import time
import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data,Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv,GATConv,GCNConv,global_mean_pool, global_add_pool, global_max_pool
import os
from torch_geometric.loader import DataLoader

import sys
import os
current_file_path = os.path.abspath(__file__)
glam_qa_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, glam_qa_dir)
from my_code.model import GraphSAGE
from tqdm import tqdm
import argparse

class CustomDataset(Dataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        """
        Args:
            root (str): The root directory where the `.pt` files are stored.
        """
        super().__init__(root, transform, pre_transform)
        self.file_list = sorted(
            [f for f in os.listdir(self.root) if f.endswith('.pt')],
            key=lambda x: int(os.path.splitext(x)[0])  # Sort by numeric order
        )

    def len(self):
        """Return the number of `.pt` files in the dataset."""
        return len(self.file_list)

    def get(self, idx):
        """
        Args:
            idx (int): Index of the file to retrieve.
        Returns:
            Data: The loaded data object from the `.pt` file.
        """
        file_path = os.path.join(self.root, self.file_list[idx])
        data = torch.load(file_path)
        if not isinstance(data, Data):
            raise ValueError(f"File {file_path} does not contain a valid Data object.")
        return data


def my_collate_fn(data_list):
    
    batch_graphID = torch.tensor([data.graphID for data in data_list], dtype=torch.long)

    batch_num_nodes = torch.tensor([data.num_nodes for data in data_list], dtype=torch.long)


    batch_indices = torch.cat([
        torch.full((data.num_nodes,), i, dtype=torch.long) for i, data in enumerate(data_list)
    ])
    ptr = torch.tensor([0] + batch_num_nodes.cumsum(dim=0).tolist())  

    input_ids = []
    target_ids = []
    attention_mask = []
    is_node = []
    for i, entry in enumerate(data_list):
            input_ids.append(entry['input_ids'])
            target_ids.append(entry['target_ids'])
            attention_mask.append(entry['attention_mask'])
            is_node.append(entry['is_node'])
        
    batch_input_ids = torch.cat(input_ids, dim=0) # tensor
    batch_target_ids = torch.cat(target_ids, dim=0) # tensor
    batch_attn_mask= torch.cat(attention_mask, dim=0) # tensor
    batch_is_node = torch.cat(is_node, dim=0) # tensor

    

    batch_data = Data(
        input_ids = batch_input_ids,
        target_ids = batch_target_ids,
        attention_mask = batch_attn_mask,
        is_node = batch_is_node,
        num_nodes=batch_num_nodes,
        batch_graphID =batch_graphID,
        batch=batch_indices,
        ptr=ptr,
    )
    return batch_data





def train(train_loader,num_nodes):
    model.train()

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss / num_nodes



def format_time(elapsed_time):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return hours, minutes, seconds


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=1, help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num_hidden', type=int, default=1792)
    argparser.add_argument('--num_out', type=int, default=3584)
    argparser.add_argument('--num_layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--gnn_type', type=str, default='sage')
    
    args = argparser.parse_args()
    
    data_cls = 'PubMedQA'####MedQA-en,MedMCQA,PubMedQA
    path = f'../Data/ultra/{data_cls}/train_qa_qwen2.5_7b'

    subgraph_dataset = CustomDataset(root=path)
    train_loader = DataLoader(subgraph_dataset, batch_size=128, shuffle=True, collate_fn=my_collate_fn)

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
   
    for data in train_loader:
        num_node_features = data.x.shape[1]
        break

    print(num_node_features)
    model = GraphSAGE(
        num_node_features,
        hidden_channels=args.num_hidden,
        out_channels=args.num_out,
        n_layers=args.num_layers,
        num_proj_hidden=args.num_out,
        activation=F.relu,
        dropout=args.dropout,
        edge_dim=None,
        gnn_type=args.gnn_type
    ).to(device)
    model = model.to(dtype=torch.bfloat16)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    times = []
    loss = 0
    total_len = 0
    best_loss = 1000000000
    start_time = time.time()
    save_folder = f'../saved_output/{data_cls}/saved_gnn_lp'
    os.makedirs(save_folder, exist_ok=True)
    for epoch in range(10):
        epoch_loss = 0
        for batch_data in tqdm(train_loader, desc="Training Progress"):
            num_nodes = batch_data.num_nodes
            sub_train_loader = LinkNeighborLoader(
                batch_data,
                batch_size=65536,
                shuffle=True,
                neg_sampling_ratio=1.0,
                num_neighbors=[10, 10],
            )
            batch_loss = train(sub_train_loader,num_nodes)
            loss += batch_loss
            epoch_loss += batch_loss
            total_len += num_nodes
            
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_increase = 0

            best_model_path = os.path.join(save_folder, f'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
        
        print(f"epoch{epoch} avg loss:{loss/total_len}")
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, minutes, seconds = format_time(elapsed_time)
    print(f"Total time: {hours}h {minutes}min {seconds}s")

