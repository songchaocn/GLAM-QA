import argparse
import pprint
import yaml
from os import path

module_path = path.dirname(path.abspath(__file__))


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--project", type=str, default="project_GraphLLM")
    parser.add_argument("--exp_num", default=1)

    # Model Config
    parser.add_argument('--backbone', type=str, default='llama-v1-7b')
    parser.add_argument('--graph_pooling', type=str, default='mean')

    parser.add_argument('--model_class', type=str, default='InstructGLM')
    parser.add_argument('--gt_layers', type=int, default=2)
    parser.add_argument('--num_token', type=int, default=5)
    parser.add_argument('--head', type=int, default=2)

    parser.add_argument('--att_d_model', type=int, default=1792)
    parser.add_argument('--gnn_output', type=int, default=3584)
 
    parser.add_argument('--gnn_input', type=int, default=768)

    parser.add_argument('--max_text_length', type=int, default=700)

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--freeze_llama', action='store_true')
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--weight_decay', type=float, default=0.0)   
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default='cosine')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument("--grad_steps", type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--best_epoch', type=int, default=0)

    # Inference
    parser.add_argument('--gen_max_length', type=int, default=64)

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
