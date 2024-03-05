import torch
import logging
import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.utils import set_seed, resample_data, spike_to_counts2
from src.utils import load_mat, spike_to_counts1, save_data2txt, gaussian_nomalization
from src.model import Transformer
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, random_split, Subset
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# data
modelType = "Transormer"
dataFile = "indy_20160624_03.mat"
dataPath = "data/"
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 32
nEpoch = 1
modelLevel = "word"     # "character" or "word"
ctxLen = 128    # the length of the sequence
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


class Dataset(Dataset):
    def __init__(self, ctx_len, vocab_size, spike, target):
        print("loading data...", end=' ')
        self.ctxLen = ctx_len
        self.vocabSize = vocab_size
        self.x = spike
        self.y = target

        # Gaussian normalization
        # self.x, self.y = gaussian_nomalization(x, y)

        # min-max normalization
        # self.x, self.y = min_max_nomalization(x, y)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        # i = np.random.randint(0, len(self.x) - self.ctxLen)
        if torch.is_tensor(item):
            idx = item.tolist()
            x = self.x[idx]
            y = self.y[idx]
        else:
            i = item % (len(self.x) - self.ctxLen)
            x = torch.tensor(self.x[i:i + self.ctxLen, :], dtype=torch.float32)
            y = torch.tensor(self.y[i:i + self.ctxLen, :], dtype=torch.float32)
        # 用于测试的简化版本
        # x = torch.randn(self.ctxLen, 96)  # 假设数据形状为[ctxLen, 96]
        # y = torch.randn(self.ctxLen, 2)  # 假设标签形状为[ctxLen, 2]
        return x, y

class Dataset_list(Dataset):
    def __init__(self, ctx_len, vocab_size, x, y):
        self.ctxLen = ctx_len
        self.vocabSize = vocab_size
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx]
        y = self.y[idx]
        return x, y

def split_dataset(ctxLen, out_dim, dataset, train_size):
    test_size = len(dataset) - train_size
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, len(dataset)))

    train_x = dataset.x[train_indices]
    train_y = dataset.y[train_indices]
    save_data2txt(train_x, 'src_trg_data/train_spike_num.txt')
    save_data2txt(train_y, 'src_trg_data/train_target_velocity.txt')

    test_x = dataset.x[test_indices]
    test_y = dataset.y[test_indices]
    save_data2txt(test_x, 'src_trg_data/test_spike_num.txt')
    save_data2txt(test_y, 'src_trg_data/test_target_velocity.txt')

    train_dataset = Dataset(ctxLen, out_dim, train_x, train_y)
    test_dataset = Dataset(ctxLen, out_dim, test_x, test_y)

    return train_dataset, test_dataset

spike, y, t = load_mat(dataPath+dataFile)
# y = resample_data(y, 4, 1)
# new_time = np.linspace(t[0, 0], t[0, -1], len(y))
# spike, target = spike_to_counts2(spike, y, np.transpose(new_time), gap_num)
spike, target = spike_to_counts1(spike, y, t[0])
# spike = np.transpose(spike)

# spike = np.load('data/indy_20160622_01_processed_spike.npy')
# target = np.load('data/indy_20160622_01_processed_target.npy')

dataset = Dataset(ctxLen, out_dim, spike, target)

# 归一化
dataset.x, dataset.y = gaussian_nomalization(dataset.x, dataset.y)
# 平滑处理
dataset.x = gaussian_filter1d(dataset.x, 3, axis=0)
dataset.y = gaussian_filter1d(dataset.y, 3, axis=0)

src_pad_idx = -1
trg_pad_idx = -1
src_feature_dim = dataset.x.shape[1]
trg_feature_dim = dataset.y.shape[1]
max_length = ctxLen

# 按时间连续性划分数据集
# trainSize = int(0.8 * len(dataset))
# train_Dataset, test_Dataset = split_dataset(ctxLen, out_dim, dataset, trainSize)
train_Dataset = Subset(dataset, range(0, int(0.8 * len(dataset))))
test_Dataset = Subset(dataset, range(int(0.8 * len(dataset)), len(dataset)))

# setting the model parameters
model = Transformer(src_feature_dim, trg_feature_dim, src_pad_idx, trg_pad_idx, max_length)

print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize, 'betas', betas, 'eps', eps, 'wd', weightDecay,
      'ctx', ctxLen)

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                      warmupTokens=0, finalTokens=nEpoch*len(train_Dataset)*ctxLen, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                      out_dim=out_dim, ctxLen=ctxLen, embed_size=embed_size)
trainer = Trainer(model, train_Dataset, test_Dataset, tConf)
trainer.train()
trainer.test()

torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
           + '.pth')
