import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .layers import SAGEConv
from math import sqrt
import random
import os
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


  
class GraphEncoder(nn.Module):    # 
    def __init__(self, args,device, llama_embed):
        super(GraphEncoder, self).__init__()
        self.args = args
        self.GT = GraphSAGE(args)
        self.device = device

        self.embed_tokens = llama_embed   # freezed Llama-7b word embeddings.
        self.embed_dim = llama_embed.shape[1]

        
        self.graph_projector = nn.Sequential(
            nn.Linear(args.gnn_output, self.args.num_token * self.embed_dim),
            )
   
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)

    def forward(self, input_ids, is_node, graph,device):
        
        
        batch_size = input_ids.shape[0]
        
       
        node_embedding = self.graph_projector(self.GT(graph.x, graph.edge_index, graph.batch))
     
        node_embeddings =  node_embedding.view(-1, self.embed_dim) # [bs * 5, dim]
        inputs_embeds = self.embed_tokens[input_ids].to(device)
        
        seq_length = inputs_embeds.shape[1]
        # print('inputs_embeds.shape:',inputs_embeds.shape)
        is_node = is_node.view(batch_size, seq_length)
        # print(f"input_ids shape: {input_ids.shape}, is_node shape: {is_node.shape}")
        
        inputs_embeds[is_node] = node_embeddings.to(inputs_embeds.dtype)

        return inputs_embeds# [bsz, seq, dim]


class GraphEncoder2(nn.Module):    #
    def __init__(self, args,device, llama_embed):
        super(GraphEncoder2, self).__init__()
        self.args = args
        self.GT = GraphSAGE(args)
        self.device = device

        
        self.embed_tokens = llama_embed   # freezed word embeddings.
        self.embed_dim = llama_embed.shape[1]
        self.vocab_size = llama_embed.shape[0] #vocab_size
        
        
  
        self.n_head = 16
        self.attention_dim = self.embed_dim
        self.graph_dim = self.args.gnn_output
        self.key = self.attention_dim // self.n_head
        
        self.mapping_layer = nn.Linear(self.vocab_size - 1, 2000)
        self.reprogramming_layer = ReprogrammingLayer2(self.attention_dim, self.graph_dim, self.n_head, self.args.num_token, self.key, self.embed_dim)
  
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)

    def forward(self, input_ids, is_node, graph,device):
        
        
        batch_size = input_ids.shape[0]


        h_graph_feature = self.GT(graph.x, graph.edge_index, graph.batch)#[batch_size, embed_dim]

        
        word_embeddings = self.embed_tokens[:-1]
        source_embeddings = self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0)
        graph_inputs_llm = self.reprogramming_layer(h_graph_feature.unsqueeze(1), source_embeddings, source_embeddings)
       
        node_embedding = graph_inputs_llm.squeeze(1)#[batch_size, num_token * embed_dim]
        node_embeddings =  node_embedding.view(-1, self.embed_dim)#[bs * 5, dim]
        
        inputs_embeds = self.embed_tokens[input_ids].to(device)
        
        seq_length = inputs_embeds.shape[1]
        # print('inputs_embeds.shape:',inputs_embeds.shape)
        is_node = is_node.view(batch_size, seq_length)
        # print(f"input_ids shape: {input_ids.shape}, is_node shape: {is_node.shape}")
        
        inputs_embeds[is_node] = node_embeddings.to(inputs_embeds.dtype)

        return inputs_embeds# [bsz, seq, dim]
    




class GraphSAGE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        in_channels = args.gnn_input
        n_layers = args.gt_layers
        hidden_channels = args.att_d_model
        out_channels = args.gnn_output
        # edge_dim = args.edge_dim
        num_proj_hidden = out_channels

        gnn_conv = SAGEConv
        self.convs = nn.ModuleList()
        if n_layers > 1:
            self.convs.append(gnn_conv(in_channels, hidden_channels))
            for i in range(1, n_layers - 1):
                self.convs.append(gnn_conv(hidden_channels, hidden_channels))
            self.convs.append(gnn_conv(hidden_channels, out_channels))
        else:
            self.convs.append(gnn_conv(in_channels, out_channels))

        # non-linear layer for contrastive loss
        self.fc1 = nn.Linear(out_channels, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, out_channels)

        self.activation = F.relu

        #Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool

    # def forward(self, x, edge_index, edge_attr=None, batch=None, lp=None):
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            # x = conv(x, edge_index, edge_attr=edge_attr)
            # print(f"x shape: {x.shape}, edge_index shape: {edge_index.shape}")
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)
        
        # x = list(unbatch(x, batch))# 拆分为单图节点列表
        # xs = []
        # # assert lp.shape[0] == len(x)
        # for i in range(len(x)):
        #     # if lp[i].data.item() == True:
        #     #     xs.append((x[i][0] + x[i][1]) / 2)
        #     # else:
        #         # xs.append(x[i][0])
        #      xs.append(x[i][0])# 只取每个图的第一个节点(ego)
             
        # x = t.stack(xs, dim=0)
        x = self.pool(x, batch)

        return x

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    
    

class ReprogrammingLayer2(nn.Module):
    def __init__(self, attention_dim, graph_dim, n_heads,num_token,d_keys=None, llm_dim=None, attention_dropout=0.1):
        super(ReprogrammingLayer2, self).__init__()

        d_keys = d_keys or (attention_dim // n_heads)

        self.query_projection = nn.Linear(graph_dim, d_keys * n_heads)
        
        self.key_projection = nn.Linear(llm_dim, d_keys * n_heads)
        
        self.value_projection = nn.Linear(llm_dim, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, num_token*llm_dim)
        self.n_heads = n_heads
        
        self.dropout = nn.Dropout(attention_dropout)
        

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape 
        S, _ = source_embedding.shape 
        H = self.n_heads
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1) #（B, L, H * d_keys）
        
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding) * scale

        A = self.dropout(torch.softmax(scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
    
    
    