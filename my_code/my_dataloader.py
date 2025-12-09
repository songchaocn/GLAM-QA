
from torch_geometric.data import Dataset,Data
import os
import torch

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
    
def custom_collate_fn(data_list):

   
    edge_type = []
    for data in data_list:
        print(data.edge_type)
        edge_type.extend(data.edge_type)
    batch_edge_type = torch.tensor(edge_type, dtype=torch.long)


    
    questionID = []
    for data in data_list:
        questionID.extend(data.questionID)    
    batch_questionID = torch.tensor(questionID, dtype=torch.long)

    batch_num_nodes = torch.tensor([data.num_nodes for data in data_list], dtype=torch.long)

    
    batch_indices = torch.cat([
        torch.full((data.num_nodes,), i, dtype=torch.long) for i, data in enumerate(data_list)
    ])
    ptr = torch.tensor([0] + batch_num_nodes.cumsum(dim=0).tolist())  

 
    batch_data = Data(
        edge_type=batch_edge_type,
        questionID=batch_questionID,
        # graphID=batch_graphID,
        num_nodes=batch_num_nodes,
        batch=batch_indices,
        ptr=ptr,
    )
    return batch_data


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
