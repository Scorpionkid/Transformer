import torch
import torch.nn as nn
import logging
import datetime
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.utils import set_seed, resample_data, spike_to_counts2, load_npy, save_to_excel
from src.utils import load_mat, spike_to_counts1, save_data2txt, gaussian_nomalization, pad_sequences, parameter_search
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
modelType = "Transformer"
dataFile = "indy_20160627_01.mat"
dataPath = "../data/Makin/"
npy_folder_path = "../data/Makin_processed_npy"
ori_npy_folder_path = "../data/Makin_origin_npy"
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

# loading data
print('loading data... ' + ori_npy_folder_path)


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


# spike, y, t = load_mat(dataPath+dataFile)
# y = resample_data(y, 4, 1)
# new_time = np.linspace(t[0, 0], t[0, -1], len(y))
# spike, target = spike_to_counts2(spike, y, np.transpose(new_time), gap_num)
# spike, target = spike_to_counts1(spike, y, t[0])

# src_pad_idx = -1
# trg_pad_idx = -1
# src_feature_dim = 96
# trg_feature_dim = 96
# max_length = seq_size
# # setting the model parameters
# model = Transformer(src_feature_dim, trg_feature_dim, src_pad_idx, trg_pad_idx, max_length, embed_size, num_layers,
#                     forward_expansion, heads)
#
# from thop import profile
# input1 = torch.randn((1, 128, 128, 96))
# flops, params = profile(model, inputs=input1.to('cuda'))
# print('Macs = ' + str(flops / 1000 ** 3) + 'G')
# print('Params = ' + str(params / 1000 ** 2) + 'M')
#
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total parameters: {total_params}')

# 获取spike和target子目录的绝对路径
spike_subdir = os.path.join(ori_npy_folder_path, "spike")
target_subdir = os.path.join(ori_npy_folder_path, "target")

# 获取spike和target目录下所有的npy文件名
spike_files = sorted([f for f in os.listdir(spike_subdir) if f.endswith('.npy')])
target_files = sorted([f for f in os.listdir(target_subdir) if f.endswith('.npy')])

# 确保两个目录下的文件一一对应
assert len(spike_files) == len(target_files)
results = []

# 遍历文件并对每一对spike和target文件进行处理
for spike_file, target_file in zip(spike_files, target_files):
    # 提取前缀名以确保对应文件正确
    prefix = spike_file.split('_spike')[0]
    prefixes = [
        'indy_20161005_06',
        # 'indy_20160921_01',
        # 'indy_20160927_06',
        # 'indy_20160927_04',
        # 'indy_20161024_03',
        # 'indy_20160915_01',
        # 'indy_20160930_05',
        # 'indy_20161220_02',
        # 'indy_20161207_02',
        # 'indy_20161025_04',
        # 'indy_20161007_02',
        # 'indy_20160916_01',
        # 'indy_20160930_02',
        # 'indy_20161017_02',
        # 'indy_20161026_03',
        # 'indy_20161013_03',
        # 'indy_20161006_02',
        # 'indy_20161212_02',
        # 'indy_20161014_04',
        # 'indy_20161027_03',
        # 'indy_20170123_02',
        # 'indy_20160624_03',
        # 'indy_20161011_03',
        # 'indy_20161206_02',
        # 'indy_20170124_01',
        # 'indy_20170131_02',
        # 'indy_20170127_03'
    ]
    if prefix not in prefixes:
        continue

    assert prefix in target_file, f"Mismatched prefix: {prefix} vs {target_file}"

    # 加载spike和target的npy文件
    spike = np.load(os.path.join(spike_subdir, spike_file))
    target = np.load(os.path.join(target_subdir, target_file))

    # 计算分割点
    split_idx = int(len(spike) * 0.8)

    # 分割数据
    spike_train, spike_test = spike[:split_idx], spike[split_idx:]
    target_train, target_test = target[:split_idx], target[split_idx:]

    # 初始化数据集
    train_dataset = Dataset(seq_size, out_size, spike_train, target_train, train_mode=True)
    test_dataset = Dataset(seq_size, out_size, spike_test, target_test, train_mode=False)

    src_pad_idx = -1
    trg_pad_idx = -1
    src_feature_dim = 96
    trg_feature_dim = 96
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
                          warmupTokens=0, finalTokens=nEpoch * len(train_dataset) * seq_size, numWorkers=0,
                          epochSaveFrequency=epochSaveFrequency, epochSavePath=epochSavePath,
                          out_dim=out_size, ctxLen=seq_size, embed_size=embed_size, criterion=criterion)

    trainer = Trainer(model, train_dataset, test_dataset, tConf)
    trainer.train()
    result = trainer.test()

    # 参数搜索，备用
    # parameter_search(embed_sizes, num_layers_list, forward_expansions, heads_list, src_feature_dim, trg_feature_dim,
    #                  src_pad_idx, trg_pad_idx, max_length,
    #                  train_dataset, test_dataset, tConf,
    #                  excel_path, modelType, nEpoch, prefix)

    result['file_name'] = prefix
    results.append(result)
    torch.save(model, epochSavePath + trainer.get_runName() +
    '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
    print(prefix + 'done')
save_to_excel(results, excel_path + os.path.basename(ori_npy_folder_path) + '-' + modelType + '-' + str(nEpoch) +
              '-' + 'partialResults.xlsx', modelType, nEpoch, dimensions)
