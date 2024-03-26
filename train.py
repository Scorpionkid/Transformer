import torch
import torch.nn as nn
import logging
import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.utils import set_seed, resample_data, spike_to_counts2
from src.utils import load_mat, spike_to_counts1, save_data2txt, gaussian_nomalization, pad_sequences
from src.model import Transformer
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, random_split, Subset
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['NUMEXPR_MAX_THREADS'] = '16'




set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "Transormer"
dataFile = "indy_20160627_01.mat"
dataPath = "../data/Makin/"
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 32
nEpoch = 50
modelLevel = "word"     # "character" or "word"
seq_size = 128    # the length of the sequence
out_size = 2   # the output dim
embed_size = 256

# learning rate
lrInit = 6e-4 if modelType == "Transormer" else 4e3   # Transormer can use higher learning rate
lrFinal = 4e-4

betas = (0.9, 0.99)
eps = 4e-9
weightDecay = 0 if modelType == "Transormer" else 0.01
epochLengthFixed = 10000    # make every epoch very short, so we can see the training progress


# loading data
print('loading data... ' + dataFile)


class Dataset(Dataset):
    def __init__(self, seq_size, out_size, spike, target, train_mode=True):
        print("loading data...", end=' ')
        self.seq_size = seq_size
        self.out_size = out_size
        self.x = spike
        self.y = target
        self.train_mode = train_mode

        # Gaussian normalization
        # self.x, self.y = gaussian_nomalization(x, y)

        # min-max normalization
        # self.x, self.y = min_max_nomalization(x, y)


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

spike, y, t = load_mat(dataPath+dataFile)
# y = resample_data(y, 4, 1)
# new_time = np.linspace(t[0, 0], t[0, -1], len(y))
# spike, target = spike_to_counts2(spike, y, np.transpose(new_time), gap_num)
spike, target = spike_to_counts1(spike, y, t[0])

# 计算分割点
split_idx = int(len(spike) * 0.8)

# 分割数据
spike_train, spike_test = spike[:split_idx], spike[split_idx:]
target_train, target_test = target[:split_idx], target[split_idx:]

# 初始化数据集
train_dataset = Dataset(seq_size, out_size, spike_train, target_train, train_mode=True)
test_dataset = Dataset(seq_size, out_size, spike_test, target_test, train_mode=False)

# 归一化
train_dataset.x, train_dataset.y = gaussian_nomalization(train_dataset.x, train_dataset.y)
test_dataset.x, test_dataset.y = gaussian_nomalization(test_dataset.x, test_dataset.y)
# 平滑处理
# train_dataset.x = gaussian_filter1d(train_dataset.x, 3, axis=0)
# test_dataset.x = gaussian_filter1d(test_dataset.x, 3, axis=0)
# train_dataset.y = gaussian_filter1d(train_dataset.y, 3, axis=0)
# test_dataset.y = gaussian_filter1d(test_dataset.y, 3, axis=0)

src_pad_idx = -1
trg_pad_idx = -1
src_feature_dim = train_dataset.x.shape[1]
trg_feature_dim = train_dataset.x.shape[1]
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
                      out_dim=out_size, ctxLen=seq_size, embed_size=embed_size, criterion=criterion)
trainer = Trainer(model, train_dataset, test_dataset, tConf)
trainer.train()
trainer.test()

torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
           + '.pth')
