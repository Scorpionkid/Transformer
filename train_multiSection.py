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
dataPath = "../Makin/Makin_processed_npy/"
csv_file = 'result/train_multiSection.csv'
dataFileCoding = "utf-8"
# use 0 for char-level english and 1 for chinese. Only affects some Transormer hyperparameters
dataFileType = 0

# hyperparameter
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth-trained-"
batchSize = 32
nEpoch = 50
modelLevel = "word"     # "character" or "word"
seq_size = 1024    # the length of the sequence
out_size = 2   # the output dim
embed_size = 64
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
with open(csv_file, "a", encoding="utf-8") as file:
    file.write(dataPath + "\n")
    file.write("batch size " + str(batchSize) + "epoch  " + str(nEpoch) + "sequence len  " + str(seq_size) + "\n")


class Dataset(Dataset):
    def __init__(self, spike, target, seq_size, out_size, train_mode=True):
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


spike_trainList, spike_testList, target_trainList, target_testList, section_name = AllDays_split(dataPath)
train_spike = np.concatenate(spike_trainList, axis=0)
test_spike = np.concatenate(spike_testList, axis=0)
train_target = np.concatenate(target_trainList, axis=0)
test_target = np.concatenate(target_testList, axis=0)

train_dataset = Dataset(train_spike, train_target, seq_size, out_size)
test_dataset = Dataset(test_spike, test_target, seq_size, out_size)

src_pad_idx = -1
trg_pad_idx = -1
src_feature_dim = train_dataset.x.shape[-1]
trg_feature_dim = train_dataset.x.shape[-1]
max_length = seq_size

train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, num_workers=numWorkers, batch_size=batchSize)
test_dataloader = DataLoader(test_dataset, shuffle=False, pin_memory=True, num_workers=numWorkers, batch_size=batchSize)

model = Transformer(src_feature_dim, trg_feature_dim, src_pad_idx, trg_pad_idx, max_length)

print("number of parameters: " + str(sum(p.numel() for p in model.parameters())) + "\n")

criterion = nn.MSELoss()

print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize, 'betas', betas, 'eps', eps, 'wd', weightDecay,
      'seq_size', seq_size)

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                      warmupTokens=0, finalTokens=nEpoch*len(train_dataset)*seq_size, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                      out_dim=out_size, ctxLen=seq_size, embed_size=embed_size, criterion=criterion, csv_file=csv_file)

trainer = Trainer(model, None, None, tConf)
trainer.train(train_dataloader)

with open(csv_file, 'a', encoding='utf-8') as file:
    file.write(f"section name, test loss, test r2 score\n")

trainer.test(test_dataloader, "Test ALL")

for i, (spike_test, target_test) in enumerate(zip(spike_testList, target_testList)):
    testDataset = Dataset(spike_test, target_test, seq_size, out_size)
    testLoader = DataLoader(testDataset, shuffle=False, pin_memory=True, num_workers=numWorkers,
                            batch_size=len(testDataset))

    trainer.test(testLoader, section_name[i])

# torch.save(model, epochSavePath + trainer.get_runName() + '-' +
#            datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
