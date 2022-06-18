import torch
import numpy as np
import random
import os
import argparse
import logging

from loss import *
from utils import *
from swin import *
from transformers import get_linear_schedule_with_warmup


def train(args):

    device = args.device
    dataloader_params = {
        'batch_size': args.batch_size,
        'shuffle': args.if_shuffle, 
    }


    num_images, all_ids = calu_all_len(args.ids_dir, args.label_dir, args.seq_epochs)
    dataset = Sleepdataset(all_ids)
    
    logging.info("Init nn...")
    model = SwinTransformer(seq_epochs=args.seq_epochs,
                            img_size=(args.img_size*args.seq_epochs, 1),
                            patch_size=(args.patch_size, 1), 
                            in_chans=args.in_chans, num_classes=args.num_classes, 
                            embed_dim=args.embed_dim, # 96
                            depths=args.depths,
                            num_heads=args.num_heads,
                            window_size=args.window_size, # 
                            # ...
                            ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    training_imgaes = num_images * (args.splits-1) / args.splits
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(training_imgaes*args.num_epochs*args.warmup_portion)\
                                                    , num_training_steps=training_imgaes*args.num_epochs)
    criterion = torch.nn.CrossEntropyLoss()

    if not args.if_first_train:
        params_dict = torch.load(args.model_path)
        model.load_state_dict(params_dict['Swin'])
        optimizer.load_state_dict(params_dict['AdamW'])

    logging.info("Training...")
    cross_validate(optimizer, scheduler, criterion, model, device, args.check_step, args.baseline, \
                args.num_classes, all_ids, dataset, args.splits, args.num_epochs, dataloader_params, \
                args.data_dir, args.label_dir, args.in_chans, args.img_size, args.seq_epochs, \
                    args.batch_size)
    
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False

def main():
    logging.basicConfig(
    filename='./logs/SWIN.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])

    args.device = torch.device("cuda:0")

    args.seed = 42
    set_seed(args.seed)

    args.if_first_train = True
    args.model_path = None
    args.ids_dir = './994.txt'
    args.data_dir = '../ViT/tmp/data/'
    args.label_dir = '../ViT/tmp/labels/'
    args.seq_epochs = 3 # continuous epochs

    args.check_step = 100
    args.num_epochs = 10
    args.splits = 5

    args.img_size = 6000
    args.patch_size = 10
    args.in_chans = 7
    args.num_classes = 5
    args.embed_dim = 128
    args.depths = [2, 2, 6, 2]
    args.num_heads = [4, 8, 16, 16]
    args.window_size = (15, 1)

    args.batch_size = 32
    args.if_shuffle = True
    args.learning_rate = 2e-6
    args.warmup_portion = 0.05
    args.weight_decay = 0.01
    args.baseline = 1e7
    
    train(args)

if __name__=='__main__':
    main()