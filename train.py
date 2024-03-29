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
batchSize = 128
nEpoch = 30
modelLevel = "word"     # "character" or "word"
seq_size = 256    # the length of the sequence
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
dimensions = ['test_r2', 'test_loss', 'train_r2', 'train_loss']

# loading data
print('loading data... ' + dataFile)


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


dataset = Dataset(dataPath, seq_size, out_size)
train_dataset = Subset(dataset, range(0, int(len(dataset) * 0.8)))
test_dataset = Subset(dataset, range(int(len(dataset) * 0.8), len(dataset)))

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batchSize, pin_memory=True, num_workers=numWorkers)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset), pin_memory=True, num_workers=numWorkers)

src_pad_idx = -1
trg_pad_idx = -1
src_feature_dim = dataset.x.size(-1)
trg_feature_dim = dataset.x.size(-1)
max_length = seq_size

# 按时间连续性划分数据集
# train_Dataset = Subset(dataset, range(0, int(0.8 * len(dataset))))
# test_Dataset = Subset(dataset, range(int(0.8 * len(dataset)), len(dataset)))

# setting the model parameters
model = Transformer(src_feature_dim, trg_feature_dim, src_pad_idx, trg_pad_idx, max_length)

criterion = nn.MSELoss()

print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize, 'betas', betas, 'eps', eps, 'wd', weightDecay,
        'seq_size', seq_size)

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                        learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                        warmupTokens=0, finalTokens=nEpoch*len(train_dataset)*seq_size, numWorkers=0,
                        epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                        out_dim=out_size, ctxLen=seq_size, embed_size=embed_size, criterion=criterion, input_size=input_size)
trainer = Trainer(model, train_dataloader, test_dataloader, tConf)
trainer.train()
trainer.test()
# result['file_name'] = prefix
# results.append(result)
# torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
#             + '.pth')
# print(prefix + 'done')
# save_to_excel(results, excel_path + os.path.basename(npy_folder_path) + '-' + modelType + '-'  + str(nEpoch) + '-' + 'results.xlsx', modelType, nEpoch, dimensions)
