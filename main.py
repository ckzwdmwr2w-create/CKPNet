
import argparse
import csv, os
import random
import numpy as np
from torch import optim
import torch
import os
from data import EssayDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import main_model
#from modelGNN import main_model
#from modelGCN import main_model
#from model_aba import main_model
#from models.XAES import main_model
#from models.xlm import main_model
#from models.LaBSE import main_model
#from models.MiniLM import main_model
#from models.InfoXLM import main_model
#from models.LLM import main_model
from train import trainer
import warnings
warnings.filterwarnings("ignore")

def set_seed(args): 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
   
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']

parser = argparse.ArgumentParser(description='Essay Scoring')
parser.add_argument('--data_path', type = str, default = './dataset')
parser.add_argument('--src_data', type = list, default = ['cs_merlin', 'de_merlin', 'es_cedel2', 'en_write_and_improve', 'it_merlin'])
parser.add_argument('--tgt_data', type = list, default = ['pt_cople2'])
#parser.add_argument('--data_path', type = str, default = './data_translation')
#parser.add_argument('--src_data', type = list, default = ['es_cedel2'])    
#parser.add_argument('--tgt_data', type = list, default = ['en_es'])
parser.add_argument('--device', type = str, default ='cuda:1')
parser.add_argument('--Encoder', type = str, default = "bert-base-multilingual-uncased")
#parser.add_argument('--model', type = str, default ="/home/shiman/models/Qwen3-0.6B")
#parser.add_argument('--model', type = str, default ="/home/shiman/models/Llama-3.2-3B-Instruct")
parser.add_argument('--batch_size', type = int, default = 16) 
parser.add_argument('--n_epochs', type = int, default = 20) 
parser.add_argument('--hidden_dim', default = 768, type = int) 
parser.add_argument('--lr', type = float, default = 1e-5) # de 的lr: 5e-5
#parser.add_argument("--num_warmup_steps", default = 100, type = float)
#parser.add_argument('--seed', type = int, default = 42) # 42,  172,  610,  3, 548, 107, 615, 436, 

def load_data(data_path, data_set):
    #print('Loading dataset ...')
    examples = []
    for name in data_set:
        fp_dataset = os.path.join(data_path, name + '.tsv')
        with open(fp_dataset, "r", encoding = "utf-8") as f:
            tsv_reader = csv.DictReader(f, delimiter = "\t")
            tsv_reader = list(tsv_reader)
            #print(f"{name}: {len(list(tsv_reader))}")
            for item in tsv_reader:
                examples.append({
                    "essay": item["essay"],
                    "score": float(item["essay_score"]),
                })
    return examples

def main():
    args = parser.parse_args()
    assert len(args.src_data) == 5 and len(args.tgt_data) == 1
    print(args.Encoder)
    #args.seed = random.randint(1, 1000)
    args.seed =  42
    set_seed(args) 
    print('seed: ', args.seed)
    print('lr: ', args.lr)
    print('src_data: ', args.src_data)
    print('tgt_data: ', args.tgt_data)
    # construct data
    all_data = load_data(args.data_path, args.src_data)
    train_examples, dev_examples = train_test_split(
        all_data,
        test_size = 0.25,
    )
    test_examples = load_data(args.data_path, args.tgt_data)
    #
    train_data = EssayDataset(dataset = train_examples)
    val_data = EssayDataset(dataset = dev_examples)
    test_data = EssayDataset(dataset = test_examples)
    # load data
    train_loader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle = True, drop_last = True)
    val_loader = DataLoader(dataset = val_data, batch_size = args.batch_size, shuffle = True, drop_last = True)
    test_loader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = True, drop_last = True)
    print('train_data: ',   len(train_data),   '   train_loader: ',   len(train_loader))
    print('val_data: ',   len(val_data),   '   val_loader: ',   len(val_loader))
    print('test_data: ',  len(test_data), '    test_loader: ',    len(test_loader))
    model = main_model(args)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

    from transformers import get_linear_schedule_with_warmup
    num_training_steps = args.n_epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0.05 * num_training_steps, num_training_steps = num_training_steps)
    print('num_training_steps: ', num_training_steps)

    print("Training...")
    trainer(args = args,
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        model = model,
        optimizer = optimizer,
        scheduler = lr_scheduler
        )
    
if __name__ == '__main__':
    main()








