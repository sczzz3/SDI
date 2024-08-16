
import os
import numpy as np
import random
import logging
import pickle
import configparser
import argparse

import torch 
from torch.utils.data import IterableDataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

from loss import PairMarginRankLoss
from model import Net


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PSGDataSet(IterableDataset):
    def __init__(self, data_folder, cohorts):
        super().__init__()
        self.folder = data_folder
        self.data_files = []
        for c in cohorts:
            with open(f"../data_index/{c}_train.pkl", 'rb') as f:
                data = pickle.load(f)
                self.data_files += [os.path.join(self.folder, c, x) for x in data]
        random.shuffle(self.data_files)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_data_files = self.data_files
        else:
            per_worker = int(np.ceil(len(self.data_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data_files))
            iter_data_files = self.data_files[iter_start:iter_end]

        for file_name in iter_data_files:
            data = np.load(file_name, mmap_mode='r')
            psg = torch.from_numpy(data['psg']).float()
            stage_label = torch.tensor(data['y']).long()[:, 0]
            nerm_label = torch.zeros_like(stage_label)
            nerm_label[stage_label <= 3] = 0
            nerm_label[stage_label == 4] = 1

            perm = torch.randperm(psg.size(0))
            permuted_psg = psg.index_select(0, perm)
            permuted_labels = stage_label[perm]
            permuted_nerm = nerm_label[perm]
            for c in range(permuted_psg.shape[0]):
                yield permuted_psg[c], permuted_labels[c], permuted_nerm[c]

def run(model, data_folder, cohorts, rank_loss, n_epoch, lr, decay, batch_size, device, checkup_steps, warmup_steps, num_train_steps, save_path):

    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': lr, 'weight_decay': decay}])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    rem_loss = torch.nn.CrossEntropyLoss()

    model.train()
    train_loss = 0
    train_steps = 0
    logging.info('==========================================')
    logging.info('              Start Training              ')
    logging.info('==========================================')

    for epoch in range(n_epoch):
        dataset = PSGDataSet(data_folder, cohorts)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=64, pin_memory=True)

        model.train()
        for step, batch in enumerate(dataloader):
            train_steps += 1
            psg, label, nerm = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True), batch[2].to(device, non_blocking=True)

            pred_depth, pred_nerm = model(psg)
            if pred_depth.shape[0] == 1:
                continue

            loss = rank_loss(pred_depth, label)
            loss += rem_loss(pred_nerm, nerm)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            scheduler.step()

            if train_steps % checkup_steps == 0:
                logging.info('[%d, %5d] loss: %.10f' % (epoch, train_steps, train_loss / train_steps))

        torch.save(model.to('cpu').state_dict(), os.path.join(save_path, 'epoch'+str(epoch)+'_ckpt.pt'))
        model.to(device)
        logging.info('===================================')


def main(config_path):
    
    config = configparser.ConfigParser()
    config.read(config_path)

    set_seed(int(config['TRAINING']['seed']))
    device = torch.device(config['TRAINING']['device'])

    data_folder = config['DATA']['data_folder']
    save_path = config['DATA']['save_path']
    cohorts = config['DATA']['cohorts'].split(',')

    n_epochs = int(config['TRAINING']['n_epochs'])
    lr = float(config['TRAINING']['lr'])
    weight_decay = float(config['TRAINING']['weight_decay'])
    batch_size = int(config['TRAINING']['batch_size'])
    checkup_steps = int(config['TRAINING']['checkup_steps'])
    warmup_steps = int(config['TRAINING']['warmup_steps'])
    num_train_steps = int(config['TRAINING']['num_train_steps'])

    model = Net().to(device)
    rank_loss = PairMarginRankLoss().to(device)

    logging.basicConfig(
        filename=config['LOGGING']['log_file'],
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s'
    )

    run(model, data_folder, cohorts, rank_loss, n_epochs, lr, weight_decay, batch_size, device, checkup_steps, warmup_steps, num_train_steps, save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a model using a config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file.")
    
    args = parser.parse_args()
    main(args.config)
