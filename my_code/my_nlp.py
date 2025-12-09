
from tqdm import tqdm
import torch
from torch import nn
from transformers import  GenerationConfig

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.")


generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    max_new_tokens=800
)



def output_decode(eval_output, tokenizer):
    eval_decode_output = []

    for i in range(len(eval_output)):
        batch_output = eval_output[i]

        eval_decode_output.extend(tokenizer.batch_decode(batch_output, skip_special_tokens=True))

    return eval_decode_output


def output_decode2(eval_output, eval_label, tokenizer):
    eval_decode_output = []
    eval_decode_label = []
    assert len(eval_output) == len(eval_label)
    for i in range(len(eval_output)):
        batch_output = eval_output[i]
        label_output = eval_label[i]
        eval_decode_output.extend(tokenizer.batch_decode(batch_output, skip_special_tokens=True))
        eval_decode_label.extend(tokenizer.batch_decode(label_output, skip_special_tokens=True))
    assert len(eval_decode_label) == len(eval_decode_output)

    return eval_decode_output, eval_decode_label

