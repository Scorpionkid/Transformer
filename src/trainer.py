import math
import torch
import logging
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader
from torcheval.metrics.functional import r2_score
import matplotlib.pyplot as plt

from .utils import save_data2txt, load_npy

logger = logging.getLogger(__name__)


class TrainerConfig:
    maxEpochs = 10
    batchSize = 32
    learningRate = 4e-3
    betas = (0.9, 0.99)
    eps = 1e-8
    gradNormClip = 1.0
    weightDecay = 0.01
    lrDecay = False
    warmupTokens = 375e6
    finalTokens = 260e9
    epochSaveFrequency = 0
    epochSavePath = 'trained-'
    numWorkers = 0
    warmup_steps = 4000
    total_steps = 10000

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.Loss_train = []
        self.r2_train = []
        self.Loss_test = []
        self.r2_test = []
        self.results = {}
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.t = False
        self.config = config
        self.avg_test_loss = 0
        self.tokens = 0  # counter used for learning rate decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("当前设备:", self.device)
        current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print("当前GPU设备的名称:", current_gpu_name)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

    def get_runName(self):
        rawModel = self.model.module if hasattr(self.model, "module") else self.model
        cfg = self.config
        runName = str(cfg.out_dim) + '-' + str(cfg.ctxLen) + '-' + cfg.modelType + '-' + str(
            cfg.embed_size)

        return runName

    def train(self):
        model, config = self.model, self.config
        rawModel = model.module if hasattr(self.model, "module") else model
        rawModel = rawModel.float()
        optimizer, scheduler = rawModel.get_optimizer_and_scheduler(config)
        loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True, batch_size=config.batchSize,
                            num_workers=config.numWorkers)

        for epoch in range(config.maxEpochs):
            predicts = []
            targets = []
            totalLoss = 0
            totalR2s = 0
            self.t = True
            model.train(True)
            pbar = tqdm(enumerate(loader), total=len(loader),
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if self.t else enumerate(loader)
            for it, (x, y) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(self.t):
                    out = model(x)
                    predicts.append(out.view(-1, 2).cpu().detach())
                    targets.append(y.view(-1, 2).cpu().detach())

                    if self.t:
                        model.zero_grad()
                        loss = self.config.criterion(out.view(-1, 2), y.view(-1, 2))
                        # loss = loss.mean()
                        r2_s = r2_score(out.view(-1, 2), y.view(-1, 2))
                        totalLoss += loss.item()
                        totalR2s += r2_s.item()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradNormClip)
                        optimizer.step()
                        scheduler.step()

                        if config.lrDecay:
                            self.tokens += (y >= 0).sum()
                            lrFinalFactor = config.lrFinal / config.learningRate
                            if self.tokens < config.warmupTokens:
                                # linear warmup
                                lrMult = lrFinalFactor + (1 - lrFinalFactor) * float(self.tokens) / float(
                                    config.warmupTokens)
                                progress = 0
                            else:
                                # cosine learning rate decay
                                progress = float(self.tokens - config.warmupTokens) / float(
                                    max(1, config.finalTokens - config.warmupTokens))
                                # progress = min(progress * 1.1, 1.0) # more fine-tuning with low LR
                                lrMult = (0.5 + lrFinalFactor / 2) + (0.5 - lrFinalFactor / 2) * math.cos(
                                    math.pi * progress)

                            lr = config.learningRate * lrMult
                            for paramGroup in optimizer.param_groups:
                                paramGroup['lr'] = lr
                        else:
                            lr = config.learningRate
                        pbar.set_description(
                            f"epoch {epoch + 1} "
                            # f"progress {progress * 100.0:.2f}%"
                            f"iter {it + 1}: r2_score "
                            f"{totalR2s / (it + 1):.2f} loss {totalLoss / (it + 1):.4f}"
                            f"lr {optimizer.param_groups[0]['lr']:e}")
            # 画图就用每个epoch的数据
            # self.Loss_train.append(totalLoss / (it + 1))
            # self.r2_train.append(totalR2s / (it + 1))

            if epoch == self.config.maxEpochs - 1:
            # 如果不画图就用最后一个epoch的数据存进excel中
                self.Loss_train.append(totalLoss / (it + 1))
                self.r2_train.append(totalR2s / (it + 1))
                print(f"Train Loss: {totalLoss / (it + 1):.4f}, R2_score: {totalR2s / (it + 1):.4f},  Epoch: {self.config.maxEpochs}")

    def test(self):
        model, config = self.model, self.config
        model.eval()
        data = self.test_dataset
        totalLoss = 0
        totalR2s = 0
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                            batch_size=len(data),
                            num_workers=config.numWorkers)

        pbar = tqdm(enumerate(loader), total=len(loader),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if self.t else enumerate(loader)
        for it, (x, y) in pbar:
            x = x.to(self.device)  # place data on the correct device

            with torch.no_grad():
                out = model(x)  # forward the model
                out = out.cpu().detach()
                loss = self.config.criterion(out.view(-1, 2), y.view(-1, 2))
                # loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                r2_s = r2_score(out.view(-1, 2), y.view(-1, 2))
            totalLoss += loss.item()
            totalR2s += r2_s.item()
            self.Loss_test.append(loss.item())
            self.r2_test.append(r2_s.item())
            # print(f"Batch Loss: {loss:.4f} R2_score: {r2_s:.4f}")
        it += 1
        MeanLoss = totalLoss / it
        MeanR2 = totalR2s / it
        print(f"R2_score: {MeanR2:.4f}, Test Mean Loss: {MeanLoss:.4f},  Num_iter: {it} ")
        # with open(config.csv_file, "a", encoding='utf-8') as file:
        #     file.write(f"{section_name}, {MeanLoss:.4f}, "
        #                f"{self.config.modelType},"
        #                f"{MeanR2:.4f}, \n")

        self.results['test_loss'] = MeanLoss
        self.results['test_r2'] = MeanR2
        self.results['train_loss'] = self.Loss_train[-1]
        self.results['train_r2'] = self.r2_train[-1]

        return self.results

        n = 10000
        tar = torch.cat(targets, dim=0).cpu().detach().numpy()
        pre = torch.cat(predicts, dim=0).cpu().detach().numpy()
        tar_x_v = tar[:n, 0]
        tar_y_v = tar[:n, 1]
        pre_x_v = pre[:n, 0]
        pre_y_v = pre[:n, 1]

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))

        axs[0, 0].plot(range(0, self.config.maxEpochs), self.Loss_train)
        axs[0, 0].set_title("loss_train")
        axs[0, 1].plot(range(0, self.config.maxEpochs), self.r2_train)
        axs[0, 1].set_title("r2_train")

        # axs[1,0].plot(range(0, it), self.Loss_test)

        # axs[1,1].plot(range(0, it), self.r2_test)

        axs[1, 0].plot(range(0, len(tar_x_v)), tar_x_v, label='tar_x_v')
        axs[1, 0].plot(range(0, len(pre_x_v)), pre_x_v, label='pre_x_v')
        axs[1, 0].set_title(f"Loss_test\nTest loss: {MeanLoss:.4f}")
        axs[1, 0].legend()  # 调用特定轴的legend方法

        axs[1, 1].plot(range(0, len(tar_y_v)), tar_y_v, label='tar_y_v')
        axs[1, 1].plot(range(0, len(pre_y_v)), pre_y_v, label='pre_y_v')
        axs[1, 1].set_title(f"r2_test\nTest r2s: {MeanR2:.4f}")
        axs[1, 1].legend()  # 调用特定轴的legend方法

        # 自动调整子图间距
        plt.tight_layout()
        plt.show()

