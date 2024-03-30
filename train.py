import torch
import torch.nn as nn
import logging
import datetime
import numpy as np

from src.utils import *
from src.model import Transformer
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, DataLoader, Subset
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['NUMEXPR_MAX_THREADS'] = '16'




set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "Transormer"
dataFile = "Makin"
dataPath = "../Makin/Makin_origin_npy/"
excel_path = 'results/'
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 32
nEpoch = 3
modelLevel = "word"     # "character" or "word"
seq_size = 128    # the length of the sequence
out_size = 2   # the output dim
embed_size = 256
input_size = 96

# learning rate
lrInit = 6e-4 if modelType == "Transormer" else 4e3   # Transormer can use higher learning rate
lrFinal = 4e-4
numWorkers = 0

betas = (0.9, 0.99)
eps = 4e-9
weightDecay = 0 if modelType == "Transormer" else 0.01
epochLengthFixed = 10000    # make every epoch very short, so we can see the training progress

# loading data
print('loading data... ' + dataFile)


class Dataset(Dataset):
    def __init__(self, spike, target, seq_size, out_size, train_mode=True):
        print("loading data...", end=' ')
        self.seq_size = seq_size
        self.out_size = out_size
        self.x, self.y = spike, target
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
            x = pad_sequences(x_tensor, -1, self.seq_size)
            y = pad_sequences(y_tensor, -1, self.seq_size)
            # x = x_padded.clone().detach().requires_grad_(requires_grad)
            # y = y_padded.clone().detach().requires_grad_(requires_grad)
        else:
            x = torch.tensor(self.x[start_idx:end_idx, :], dtype=torch.float32)
            y = torch.tensor(self.y[start_idx:end_idx, :], dtype=torch.float32)
        return x, y


spike_train, spike_test, targte_train, target_test = AllDays_split(dataPath)

train_dataset = Dataset(spike_train, targte_train, seq_size, out_size)
test_dataset = Dataset(spike_test, target_test, seq_size, out_size)

src_pad_idx = -1
trg_pad_idx = -1
src_feature_dim = train_dataset.x.shape[-1]
trg_feature_dim = train_dataset.x.shape[-1]
max_length = seq_size

train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, num_workers=numWorkers, batch_size=batchSize)
test_dataloader = DataLoader(test_dataset, shuffle=False, pin_memory=True, num_workers=numWorkers, batch_size=len(test_dataset))

model = Transformer(src_feature_dim, trg_feature_dim, src_pad_idx, trg_pad_idx, max_length)

criterion = nn.MSELoss()

print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize, 'betas', betas, 'eps', eps, 'wd', weightDecay,
        'seq_size', seq_size)

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                        learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                        warmupTokens=0, finalTokens=nEpoch*len(train_dataset)*seq_size, numWorkers=0,
                        epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                        out_dim=out_size, ctxLen=seq_size, embed_size=embed_size, criterion=criterion)
trainer = Trainer(model, train_dataloader, test_dataloader, tConf)
trainer.train()
trainer.test()

# torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
#            + '.pth')
