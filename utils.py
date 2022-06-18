from hashlib import new
import mne
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import random
import logging
from sklearn.model_selection import KFold, StratifiedKFold 

    
def calu_all_len(ids_dir, label_dir, seq_epochs):
    all_ids = []
    all_ids_file = open(ids_dir, 'r')
    for each_id in all_ids_file:
        each_id = each_id.rstrip()
        each_id = each_id.split('/')[-1]
        all_ids.append(each_id)    

    num = len(all_ids)
    the_id = 0
    all_len = 0
    while the_id < num:

        label = np.load(label_dir + str(all_ids[the_id]) + '.npy')
        # all_len += label.shape[0] // seq_epochs
        all_len += label.shape[0] - seq_epochs + 1
        the_id += 1

    return all_len, all_ids


def load_data(data_dir, label_dir, all_ids, ids, in_chans, img_size, seq_epochs, batch_size):
    ####
    now_ids = np.array(all_ids)[ids].tolist()
    num = len(now_ids)
    ####
    the_id = 0
    while the_id < num:

        image = np.load(data_dir + str(all_ids[the_id]) + '.npy')
        label = np.load(label_dir + str(all_ids[the_id]) + '.npy')

        the_id += 1
        num_stage = image.shape[0]
        #####
        n = 0
        # num_seq = num_stage // seq_epochs # discard the end epochs
        num_seq = num_stage - seq_epochs + 1# 
        #####
        the_images = np.zeros((num_seq, in_chans, seq_epochs*img_size))
        the_labels = np.zeros((num_seq, seq_epochs))
        #####
        while n < num_seq:
            cur_image = image[n:(n+seq_epochs)]
            cur_label = label[n:(n+seq_epochs)]
            tmp_label_lst = []
            ###
            new_image = np.zeros((in_chans, seq_epochs*img_size))
            new_label = np.zeros((seq_epochs))
            ###
            for s in range(seq_epochs):
                new_image[:, s*img_size:(s+1)*img_size] = cur_image[s]
                tmp_label_lst.append(cur_label[s])
            new_label = np.array(tmp_label_lst)

            the_images[n, :, :] = new_image
            the_labels[n, :] = new_label
            n += 1

        b = 0
        threshold_batch = num_seq // batch_size
        while b < threshold_batch:
            yield the_images[b*batch_size:(b+1)*batch_size, :, :], the_labels[b*batch_size:(b+1)*batch_size, :]
            b += 1
        ####
        if b*batch_size != num_seq:
            yield the_images[b*batch_size:, :, :], the_labels[b*batch_size:, :]


class Sleepdataset(Dataset):
    def __init__(self, patients_id):
        self.ids = patients_id

    def __getitem__(self, index):
        return self.ids[index]

    def __len__(self):
        return len(self.ids)

###################
def train(optimizer, scheduler, criterion, train_loader, model, device, cur_epoch, check_step):
    train_loss = 0
    batch_num = 0
    train_cnt = 0
    ####
    for train_iter, (inputs, labels) in enumerate(train_loader):
        batch_num += 1
        train_cnt += inputs.shape[0]

        optimizer.zero_grad()
        ##
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()
        ##
        inputs, labels = inputs.to(device), labels.to(device)

        model.train()
        outputs = model(inputs)
        #######
        seq_epochs = labels.shape[1]
        loss = 0
        for s in range(seq_epochs):
            loss += criterion(outputs[:, s, :], labels[:, s].long())
        # loss = criterion(outputs, labels)
        # loss = criterion(outputs, labels, cur_epoch)
        #######
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ###
        if train_iter % check_step == 0:
            logging.info('[%d, %5d] loss: %.5f' % (cur_epoch+1, train_cnt, train_loss / train_cnt))

    return 


def test(criterion, test_loader, model, device, cur_epoch, num_classes):
    test_loss = 0
    conf_matrix = np.zeros((num_classes, num_classes))
    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(test_loader):
            ##
            inputs = torch.from_numpy(inputs).float()
            labels = torch.from_numpy(labels).long()
            ##
            inputs, labels = inputs.to(device), labels.to(device)
            model.eval()
            outputs = model(inputs)
            #######
            seq_epochs = labels.shape[1]
            loss = 0
            for s in range(seq_epochs):
                loss += criterion(outputs[:, s, :], labels[:, s])
            # loss = criterion(outputs, labels)
            # loss = criterion(outputs, labels, cur_epoch)
            #######
            test_loss += loss.item()
            #######
            for s in range(seq_epochs):
                tmp_outputs = outputs[:, s, :]
                tmp_labels = labels[:, s]
                _, prediction = tmp_outputs.max(dim=1)

                prediction = prediction.cpu().numpy()
                tmp_labels = tmp_labels.cpu().numpy()
                for p, t in zip(prediction, tmp_labels):
                    p_int = int(p)
                    t_int = int(t)
                    conf_matrix[p_int, t_int] += 1
            #######
            # _, prediction = outputs.max(dim=1)

            # for p, t in zip(prediction, labels):
            #     conf_matrix[p, t] += 1

    TP = np.zeros((num_classes,))
    FP = np.zeros((num_classes,))
    TN = np.zeros((num_classes,))
    FN = np.zeros((num_classes,))
    SUM = np.sum(conf_matrix)
    for i in range(num_classes):
        TP[i] = conf_matrix[i, i]
        FP[i] = np.sum(conf_matrix, axis=1)[i] - TP[i]
        TN[i] = SUM + TP[i] - np.sum(conf_matrix, axis=1)[i] - np.sum(conf_matrix, axis=0)[i]
        FN[i] = np.sum(conf_matrix, axis=0)[i] - TP[i]
    accuracy = (TP + TN) / SUM
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)

    logging.info("@@@@@@@@@@@ Validation @@@@@@@@@@@@")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Sepci: {specificity}")
    logging.info(f"Recall: {sensitivity}")
    logging.info(f"Precision: {precision}")
    logging.info(f"F1: {f1}")
    logging.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    return test_loss, [accuracy, specificity, sensitivity, precision, f1]


def cross_validate(optimizer, scheduler, criterion, model, device, check_step, baseline, \
                            num_classes, all_ids, dataset, splits, epochs, dataloader_params,\
                    data_dir, label_dir, in_chans, img_size, seq_epochs, batch_size):
    """
    Does cross validation for a model.
    @param model: An instance of a model to be evaluated.
    @param dataset: A torch.utils.data.Dataset dataset.
    @param splits: The number of cross validation folds.
    @param epochs: The number of epochs per cross validation fold.
    @dataloader_params: parameters to be passed to the torch.utils.data.DataLoader class.
    """
    skf = KFold(n_splits=splits)
    fold = 0

    kfold_testing_metrics = []
    for train_idx, test_idx in skf.split(dataset.ids):
        logging.info("Cross validation fold %d" %fold)

        model.reset_all_weights()

        for epoch in range(epochs):
            ####
            train_loader = load_data(data_dir, label_dir, all_ids, train_idx, in_chans, \
                                    img_size, seq_epochs, batch_size)
            test_loader = load_data(data_dir, label_dir, all_ids, test_idx, in_chans, \
                                    img_size, seq_epochs, batch_size)
            ####
            train(optimizer, scheduler, criterion, train_loader, model, device, epoch, check_step)
            test_loss, metrics = test(criterion, test_loader, model, device, epoch, num_classes)

            if test_loss < baseline:
                baseline = test_loss
                filename = './weights/SWIN.pt'
                state = {'Swin': model.state_dict(), "AdamW": optimizer.state_dict()}
                torch.save(state, filename)

        kfold_testing_metrics += metrics
        fold += 1

    logging.info("\n########################################")
    logging.info("Cross validation with %d folds complete." % splits)

    ####
    kfold_testing_metrics = np.array(kfold_testing_metrics) / splits
    accuracy = kfold_testing_metrics[0]
    specificity = kfold_testing_metrics[1]
    sensitivity = kfold_testing_metrics[2]
    precision = kfold_testing_metrics[3]
    f1 = kfold_testing_metrics[4]

    logging.info("@@@@@@@@@@@ Averaged Metrics @@@@@@@@@@@@")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Sepci: {specificity}")
    logging.info(f"Recall: {sensitivity}")
    logging.info(f"Precision: {precision}")
    logging.info(f"F1: {f1}")
    logging.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


