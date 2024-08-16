
import os
import random
import pickle

data_path = "../proc_data"
cohort = ['mesa', 'cfs', 'mros']
for c in cohort:

    c_path = os.path.join(data_path, c)
    data_files = os.listdir(c_path)

    random.seed(2024)
    random.shuffle(data_files)
    split_index = int(len(data_files) * 0.7)

    train_files = data_files[:split_index]
    test_files = data_files[split_index:]

    with open(os.path.join("../data_index/", c+'_train.pkl'), 'wb') as f:
        pickle.dump(train_files, f)
    with open(os.path.join("../data_index/", c+'_test.pkl'), 'wb') as f:
        pickle.dump(test_files, f)


