import json
import time
import torch
import os
import gc
from tqdm import tqdm
from datetime import timedelta
from torch_geometric.loader import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

import sys
import os


current_file_path = os.path.abspath(__file__)
glam_qa_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, glam_qa_dir)

from my_code import sampler, layers, my_config, conversation, language_model, graph_transformer
from my_code.my_config import *                                    
from my_code.my_dataloader import CustomDataset,my_collate_fn
from my_code.trainer_base import create_optimizer_and_scheduler
from my_code.language_model import InstructGLM2
from my_code.graph_transformer import GraphEncoder2
from my_code.my_nlp import *


def train(first_model, llm, train_loader, optimizer, warmup_scheduler, accelerator, epoch, device):
    # save_path = save_path
    total_loss = 0
    batch_loss = 0
    batch_ = 0
    first_model.train()
    llm.train()

    for batch_data in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
        graph = batch_data.to(device)
        input_ids = batch_data['input_ids']
        is_node = batch_data['is_node']
        labels = batch_data["target_ids"]
        attention_mask = batch_data['attention_mask']

        is_node = is_node.to(device)
        input_ids = input_ids.to(device)

        embeds = first_model(
            input_ids=input_ids,
            is_node=is_node,
            graph=graph,
            device=device
        )

        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        output = llm(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)
        loss = output['loss']
        
        batch_loss = loss.item()
        
        batch_ += 1
        
        if torch.isnan(loss) or torch.isinf(loss):
            # print(f"Loss is NaN or Inf at batch {batch_ + 1}. Skipping this batch.")
            continue
                
        
        total_loss += batch_loss


        accelerator.backward(loss)
        if accelerator.sync_gradients:
            # accelerator.clip_grad_norm_(first_model.parameters(), args.clip_grad_norm)
            accelerator.clip_grad_norm_(first_model.parameters(), 0.1)
        optimizer.step()
        warmup_scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    accelerator.print(f"Epoch {epoch+1} | Average Train Loss: {avg_loss:.4f}")




@torch.no_grad()
def test(first_model, llm, test_loader, tokenizer, accelerator, device,test_save_path):
    first_model.eval()
    llm.eval()
    all_results = []

    for batch_data in tqdm(test_loader, desc="Testing Progress"):
        batch_ID = batch_data.graphID.tolist()
        graph = batch_data.to(device)
        input_ids = batch_data['input_ids']
        is_node = batch_data['is_node']
        attention_mask = batch_data['attention_mask']
        
        is_node = is_node.to(device)
        input_ids = input_ids.to(device)


        embeds = first_model(
            input_ids=input_ids,
            is_node=is_node,
            graph=graph,
            device=device
        )
        
        attention_mask = attention_mask.to(device)
        results = llm.generate(    
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id,

        )
        results = accelerator.pad_across_processes(results, dim=1, pad_index=tokenizer.pad_token_id)
        results_gathered = accelerator.gather(results).cpu().numpy()



        pred_texts = tokenizer.batch_decode(results_gathered, skip_special_tokens=True)
        for bid, pred in zip(batch_ID, pred_texts):
            all_results.append({
                "graphID": bid,
                "generated_text": pred
            })

    if accelerator.is_main_process:
        save_path = test_save_path
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        accelerator.print(f"[✅] results saved to: {save_path}")

    return all_results



def main():

    args = parse_args()

    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with=None,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
        gradient_accumulation_steps=args.grad_steps
    )
    device = accelerator.device

    data_cls = 'PubMedQA'####MedQA-en,MedMCQA,PubMedQA
    model_path = 'your_path/llm/Qwen2.5-7B'
    


    train_dataset = CustomDataset(root=f'../Data/ultra/{data_cls}/train_qa_qwen2.5_7b')
    test_dataset = CustomDataset(root=f'../Data/ultra/{data_cls}/test_qa_qwen2.5_7b')


    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)

    

    llm = InstructGLM2.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'unk_token': '<unk>'})
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<t_patch>']})
    llm.resize_token_embeddings(len(tokenizer))
    # llm.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    llama_embeds = llm.get_input_embeddings().weight
    
    for n, p in llm.named_parameters():
        p.requires_grad_(False)

    first_model = GraphEncoder2(args, device, llama_embed=llama_embeds).to(device, dtype=torch.bfloat16)
    if not args.inference:
        first_model.GT.load_state_dict(torch.load(f'../saved_output/{data_cls}/saved_gnn_lp/best_model.pth'))

    for n, p in first_model.named_parameters():
        if n.split('.')[0] == 'GT':
            p.requires_grad_(False)
    

    optimizer, warmup_scheduler = create_optimizer_and_scheduler(args, first_model, llm, train_loader)

    first_model, llm, train_loader, optimizer, warmup_scheduler = accelerator.prepare(
        first_model, llm, train_loader, optimizer, warmup_scheduler
    )

    date_flag = time.strftime("%m%d", time.localtime())
    best_model_path = None
    num_epochs = 3
    for epoch in range(num_epochs):
        accelerator.print(f'\n=== Epoch {epoch+1}/{num_epochs} ===')
        save_folder = f'../saved_output/{data_cls}/ckpt'
        os.makedirs(save_folder, exist_ok=True)
        train(first_model, llm, train_loader, optimizer, warmup_scheduler, accelerator, epoch, device)
        
        best_model_path = os.path.join(save_folder, f'best_model_epoch{epoch+1}_{date_flag}.pth')
        if accelerator.is_main_process:
            accelerator.save(accelerator.unwrap_model(first_model).state_dict(), best_model_path)
            


    accelerator.wait_for_everyone()

    accelerator.print("\n start testing...")
    test_model = GraphEncoder2(args, device, llama_embed=llama_embeds).to(device, dtype=torch.bfloat16)
    ckpt_path = best_model_path
    test_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_model, llm, test_loader = accelerator.prepare(test_model, llm, test_loader)

    test_save_folder = f'../saved_output/{data_cls}/output_json'
    os.makedirs(test_save_folder, exist_ok=True)
    test_save_path = os.path.join(test_save_folder, f'test_pred_{date_flag}.json')
    
    
    test_results = test(
        first_model=test_model,
        llm=llm,
        test_loader=test_loader,
        tokenizer=tokenizer,
        accelerator=accelerator,
        device=device,
        test_save_path=test_save_path,
    )



if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"⏱ Total time: {h}h {m}m {s}s")
    
    
