import math
import torch
import logging
from tqdm.auto import tqdm

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
    def __init__(self, model, train_dataloader, test_dataloader, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.avg_test_loss = 0
        self.tokens = 0     # counter used for learning rate decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("当前设备:", self.device)
        current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print("当前GPU设备的名称:", current_gpu_name)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

    def get_runName(self):
        rawModel = self.model.module if hasattr(self.model, "module") else self.model
        cfg = rawModel.config
        runName = str(cfg.out_dim) + '-' + str(cfg.ctxLen) + '-' + cfg.modelType + '-' + str(
            cfg.embed_size)

        return runName

    def train_epoch(self, epoch, model, config, optimizer, scheduler):
        predicts = []
        targets = []
        totalLoss = 0
        totalR2s = 0
        model.train(True)

        pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for it, (x, y) in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.set_grad_enabled(True):
                pre, loss, r2_s = model(x, y)
                predicts.append(pre.detach().cpu().view(-1, 2))
                targets.append(y.detach().cpu().view(-1, 2))
                loss = loss.mean()
                totalLoss += loss.item()
                totalR2s += r2_s

                model.zero_grad()
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

                    pbar.set_description(
                        f"epoch {epoch+1} progress {progress * 100.0:.2f}% iter {it + 1}: r2_score "
                        f"{totalR2s / (it + 1):.2f} loss {totalLoss / (it + 1):.4f} lr {lr:e}")

        return predicts, targets

    def train(self):
        model, config = self.model, self.config
        rawModel = model.module if hasattr(self.model, "module") else model
        rawModel = rawModel.float()
        optimizer, scheduler = rawModel.get_optimizer_and_scheduler(config)

        for epoch in range(config.maxEpochs):
            predicts, targets = self.train_epoch(epoch, model, config, optimizer, scheduler)
            # print(self.avg_train_loss / len(self.train_dataset))

            if (config.epochSaveFrequency > 0 and epoch % config.epochSaveFrequency == 0) or (epoch ==
                                                                                              config.maxEpochs - 1):
                # DataParallel wrappers keep raw model object in .module
                rawModel = self.model.module if hasattr(self.model, "module") else self.model
                # torch.save(rawModel, self.config.epochSavePath + str(epoch + 1) + '.pth')

            # save the model predicts and targets every 10 epoch
            # if (epoch + 1) % config.epochSaveFrequency == 0:
            #     save_data_to_txt(predicts, targets, 'predict_character.txt', 'target_character.txt')

    def test(self):
        model, config = self.model, self.config
        model.eval()

        predicts = []
        targets = []
        totalLoss = 0
        totalR2s = 0

        pbar = enumerate(self.test_dataloader)

        for it, (x, y) in pbar:
            x = x.to(self.device)  # place data on the correct device
            y = y.to(self.device)

            with torch.set_grad_enabled(self.t):
                pre, loss, r2_s = model(x, y)  # forward the model
                predicts.append(pre.detach().cpu().view(-1, 2))
                targets.append(y.detach().cpu().view(-1, 2))
                loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus

            totalLoss += loss.item()
            totalR2s += r2_s
            print(f"Batch Loss: {loss:.4f} R2_score: {r2_s:.4f}")

        print(f"Test Loss: {totalLoss / (it + 1):.4f}, R2_score: {totalR2s / (it + 1):.4f}")

        # save_data2txt(predicts, 'src_trg_data/test_predict.txt')
        # save_data2txt(targets, 'src_trg_data/test_target.txt')
