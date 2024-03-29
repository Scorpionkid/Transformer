import torch
import logging
import datetime
import numpy as np

from src.utils import *
from src.model import Transformer
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, DataLoader, Subset
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



set_seed(42)
print(os.getcwd())
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "Transormer"
dataFile = "Makin"
dataPath = "../Makin/Makin_origin_npy/"
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 128
nEpoch = 3
modelLevel = "word"     # "character" or "word"
ctxLen = 256    # the length of the sequence
out_dim = 2   # the output dim
embed_size = 256
gap_num = 10    # the time slice

# learning rate
lrInit = 6e-4 if modelType == "Transormer" else 4e3   # Transormer can use higher learning rate
lrFinal = 4e-4

betas = (0.9, 0.99)
eps = 4e-9
weightDecay = 0 if modelType == "Transormer" else 0.01
epochLengthFixed = 10000    # make every epoch very short, so we can see the training progress


# loading data
print('loading data... ' + dataFile)


# class Dataset(Dataset):
#     def __init__(self, ctxLen, vocab_size, spike, target):
#         print("loading data...", end=' ')
#         self.vocabSize = vocab_size
#         self.ctxLen = ctxLen
#         self.x = spike
#         self.y = target
#         self.length = -(-len(self.x) // self.ctxLen)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, item):
#         start = item * self.ctxLen
#         end = start + self.ctxLen
#         if end >= len(self.x):
#             end = len(self.x) - 1
#         x = torch.tensor(self.x[start:end], dtype=torch.float32)
#         y = torch.tensor(self.y[start:end], dtype=torch.float32)

#         if len(x) < self.ctxLen:
#             x = torch.nn.functional.pad(x, (0, 0, 0, self.ctxLen - len(x)))
#             y = torch.nn.functional.pad(y, (0, 0, 0, self.ctxLen - len(y)))

#         return x, y
class Dataset(Dataset):
    def __init__(self, data_path, ctx_len, vocab_size):
        self.ctxLen = ctx_len
        self.vocabSize = vocab_size
        spike, target = AllDays_split(data_path)
        self.x, self.y = Reshape_ctxLen(spike, target, ctx_len)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y


dataset = Dataset(dataPath, ctxLen, out_dim)

src_pad_idx = -1
trg_pad_idx = -1
src_feature_dim = dataset.x.shape[2]
trg_feature_dim = dataset.y.shape[2]
max_length = ctxLen

# 按时间连续性划分数据集
# trainSize = int(0.8 * len(dataset))
# train_Dataset, test_Dataset = split_dataset(ctxLen, out_dim, dataset, trainSize)
train_Dataset = Subset(dataset, range(0, int(0.8 * len(dataset))))
test_Dataset = Subset(dataset, range(int(0.8 * len(dataset)), len(dataset)))

train_dataloader = DataLoader(train_Dataset, shuffle=True, batch_size=batchSize)
test_dataloader = DataLoader(test_Dataset, shuffle=False, batch_size=len(test_Dataset))

# setting the model parameters
model = Transformer(src_feature_dim, trg_feature_dim, src_pad_idx, trg_pad_idx, max_length)

print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize, 'betas', betas, 'eps', eps, 'wd', weightDecay,
      'ctx', ctxLen)

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                      warmupTokens=0, finalTokens=nEpoch*len(train_Dataset)*ctxLen, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                      out_dim=out_dim, ctxLen=ctxLen, embed_size=embed_size)
trainer = Trainer(model, train_dataloader, test_dataloader, tConf)
trainer.train()
trainer.test()

# torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
#            + '.pth')
