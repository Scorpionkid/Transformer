import os
import torch.nn as nn
import random
import logging
import datetime
import numpy as np
from src.utils import *
from src.model import Transformer
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import Dataset, random_split, Subset
import os

set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "Transformer"
dataFile = "indy_20160627_01.mat"
dataPath = "../data/Makin/"
npy_folder_path = "../data/Makin_processed_npy"
ori_npy_folder_path = "../data/Makin_origin_npy"
csv_file = "results/shuffleDaySequenceTrainAll.csv"
excel_path = 'results/'
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transformer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10  # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 32
nEpoch = 1
modelLevel = "word"  # "character" or "word"
seq_size = 128  # the length of the sequence
out_size = 2  # the output dim
embed_size = 64
num_layers = 3
forward_expansion = 2
heads = 2

# 参数范围
embed_sizes = [32, 64, 128]
num_layers_list = [1, 2, 3, 4]
forward_expansions = [1, 2, 3, 4]
heads_list = [1, 2, 4, 8]

# learning rate
lrInit = 6e-4 if modelType == "Transformer" else 4e3  # Transformer can use higher learning rate
lrFinal = 4e-4

betas = (0.9, 0.99)
eps = 4e-9
weightDecay = 0 if modelType == "Transformer" else 0.01
epochLengthFixed = 10000  # make every epoch very short, so we can see the training progress
dimensions = ['test_r2', 'test_loss', 'train_r2', 'train_loss']


class Dataset(Dataset):
    def __init__(self, seq_size, out_size, spike, target, train_mode=True):
        print("loading data...", end=' ')
        self.seq_size = seq_size
        self.out_size = out_size
        self.x = spike
        self.y = target
        self.train_mode = train_mode

    def __len__(self):
        return (len(self.x) + self.seq_size - 1) // self.seq_size

    def __getitem__(self, idx):
        start_idx = idx * self.seq_size
        end_idx = start_idx + self.seq_size
        requires_grad = True if self.train_mode == True else False

        # 处理最后一个可能不完整的序列
        if end_idx > len(self.x):
            # transformer中的padding
            x_tensor = torch.tensor(self.x[start_idx:len(self.x), :], dtype=torch.float32)
            y_tensor = torch.tensor(self.y[start_idx:len(self.y), :], dtype=torch.float32)
            x_padded = pad_sequences(x_tensor, -1, self.seq_size)
            y_padded = pad_sequences(y_tensor, -1, self.seq_size)
            x = x_padded.clone().detach().requires_grad_(requires_grad)
            y = y_padded.clone().detach().requires_grad_(requires_grad)
        else:
            x = torch.tensor(self.x[start_idx:end_idx, :], dtype=torch.float32)
            y = torch.tensor(self.y[start_idx:end_idx, :], dtype=torch.float32)
        return x, y


# dividing the dataset into training set and testing dataset by a ratio of 82
spike_train, spike_split_test, target_train, target_split_test, section_name = AllDays_split(dataPath)
train_length = sum(len(item) for item in spike_train)

# train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=batchSize,
#                           num_workers=numWorkers)

with open(csv_file, "a", encoding="utf-8") as file:
    file.write(dataPath + 'batch size  ' + str(batchSize) + 'epochs  ' + str(nEpoch) +
               'sequence  len  ' + str(seq_size) + '\n')

# 初始化数据集
train_dataset = Dataset(seq_size, out_size, spike_train, target_train, train_mode=True)

src_pad_idx = -1
trg_pad_idx = -1
src_feature_dim = train_dataset.x.shape[1]
trg_feature_dim = train_dataset.x.shape[1]
max_length = seq_size

# setting the model parameters
model = Transformer(src_feature_dim, trg_feature_dim, src_pad_idx, trg_pad_idx, max_length, embed_size, num_layers,
                    forward_expansion, heads)
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')

criterion = nn.MSELoss()

print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize, 'betas', betas, 'eps', eps, 'wd', weightDecay,
      'seq_size', seq_size)

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                      warmupTokens=0, finalTokens=nEpoch * train_length * seq_size, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                      out_dim=out_size, ctxLen=seq_size, embed_size=embed_size, criterion=criterion)
index = list(range(len(spike_train)))
random.shuffle(index)

for i in index:
    print(section_name[i] + "\n")
    train_dataset = Dataset(seq_size, out_size, spike_train[i], target_train[i], train_mode=True)
    trainer = Trainer(model, train_dataset, None, tConf)
    trainer.train()

with open(csv_file, "a", encoding="utf-8") as file:
    file.write(f"section, test loss, test r2 score, test time\n")

target_joint_test = np.concatenate(target_split_test, axis=0)
spike_joint_test = np.concatenate(spike_split_test, axis=0)
test_dataset = Dataset(seq_size, out_size, spike_joint_test, target_joint_test, train_mode=False)
trainer = Trainer(model, None, test_dataset, tConf)
trainer.test("ALL Test")


for i, (spike_single_test, target_single_test) in enumerate(zip(spike_split_test, target_split_test)):
    test_dataset = Dataset(seq_size, out_size, spike_single_test, target_single_test, train_mode=False)
    trainer = Trainer(model, None, test_dataset, tConf)
    trainer.test(section_name[i])
