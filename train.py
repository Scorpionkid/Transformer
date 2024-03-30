import torch
import logging
import datetime
import numpy as np

from src.utils import *
from src.model import Transformer
from src.trainer import Trainer, TrainerConfig
from torch.utils.data import Dataset, DataLoader, Subset

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



set_seed(42)
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

# hyperparameters
epochSaveFrequency = 10    # every ten epoch
epochSavePath = "pth/trained-"
batchSize = 64
nEpoch = 2
modelLevel = "word"     # "character" or "word"
ctxLen = 256    # the length of the sequence
out_dim = 2   # the output dim
embed_size = 256

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
    def __init__(self, data_path, ctx_len, vocab_size):
        print("loading data...", end=' ')
        self.ctxLen = ctx_len
        self.vocabSize = vocab_size
        spike, target = loadAllDays(data_path)
        self.x, self.y = Reshape_ctxLen(spike, target, ctx_len)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y


# load the data from .npy file (have been processed including normalization and gaussion filter)
dataset = Dataset(dataPath, ctxLen, out_dim)

train_dataset = Subset(dataset, range(0, int(len(dataset) * 0.8)))
test_dataset = Subset(dataset, range(int(len(dataset) * 0.8), len(dataset)))

train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, num_workers=numWorkers, batch_size=batchSize)
test_dataloader = DataLoader(test_dataset, shuffle=False, pin_memory=True, num_workers=numWorkers, batch_size=len(test_dataset))

src_pad_idx = -1
trg_pad_idx = -1
src_feature_dim = dataset.x.shape[-1]
trg_feature_dim = dataset.y.shape[-1]
max_length = ctxLen

# setting the model parameters
model = Transformer(src_feature_dim, trg_feature_dim, src_pad_idx, trg_pad_idx, max_length)

print('model', modelType, 'epoch', nEpoch, 'batchsz', batchSize, 'betas', betas, 'eps', eps, 'wd', weightDecay,
      'ctx', ctxLen)

tConf = TrainerConfig(modelType=modelType, maxEpochs=nEpoch, batchSize=batchSize, weightDecay=weightDecay,
                      learningRate=lrInit, lrDecay=True, lrFinal=lrFinal, betas=betas, eps=eps,
                      warmupTokens=0, finalTokens=nEpoch*len(train_dataset)*ctxLen, numWorkers=0,
                      epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                      out_dim=out_dim, ctxLen=ctxLen, embed_size=embed_size)
trainer = Trainer(model, train_dataloader, test_dataloader, tConf)
trainer.train()
trainer.test()

# torch.save(model, epochSavePath + trainer.get_runName() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
#            + '.pth')
