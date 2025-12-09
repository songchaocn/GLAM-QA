
from .language_model import *
from .graph_transformer import *
from transformers import get_scheduler
import torch
import random


 
def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW
    elif optim == 'adamax':
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer
   

def create_optimizer_and_scheduler(args,first_model, model, train_loader):
    #no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in first_model.named_parameters() if p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "weight_decay": args.weight_decay,
        }
    ]

    optimizer_class = get_optimizer(args.optim)
    optimizer = optimizer_class(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.adam_eps,
        betas=(args.adam_beta1, args.adam_beta1)) # betas=(0.9, 0.95)

    
    total_steps = len(train_loader) * args.epoch
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    assert num_warmup_steps <= total_steps, \
        'num_warmup_steps {} is too large, more than total_steps {}'.format(num_warmup_steps, total_steps)
    warmup_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_steps
        )

    return optimizer, warmup_scheduler
