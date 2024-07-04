import time
import os

import librosa
import soundfile
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
from models.DialogueRNN import BiModel
from loss import MaskedLoss, InfoNCE
from models.HybridRNN import MARN
from models.lsthm_newz import MARN1_newz


class ModelTrainer(nn.Module):

    def __init__(self, device, lr, test_step, lr_decay, model, loss, **kwargs):
        super(ModelTrainer, self).__init__()
        self.device = device
        # 定义模型
        if model == 'DialogueRNN':
            D_m = 712
            D_g = 500
            D_p = 500
            D_e = 300
            D_h = 300
            # D_1 = 100
            # D_2 = 100
            # D_3 = 512
            self.model = BiModel(D_m, D_g, D_p, D_e, D_h,
                                n_classes=6,
                                listener_state=True,
                                context_attention='general',
                                dropout_rec=0.1,
                                dropout=0.1).to(self.device)
            # self.model = Latemodal(D_1,D_2,D_3,D_m, D_g, D_p, D_e, D_h,
            #                 n_classes=6,
            #                 listener_state=True,
            #                 context_attention='general',
            #                 dropout_rec=0.1,
            #                 dropout=0.1).to(self.device)
            # self.model = Midmodal(D_1,D_2,D_3,D_m, D_g, D_p, D_e, D_h,
            #                 n_classes=6,
            #                 listener_state=True,
            #                 context_attention='general',
            #                 dropout_rec=0.1,
            #                 dropout=0.1).to(self.device)
        elif model=='MARN':
            self.model=MARN().to(self.device)
        elif model == 'MARN1_newz':
            self.model=MARN1_newz().to(self.device)
        
        # 定义损失函数
        if loss == 'CrossEntropy':
            losser = nn.CrossEntropyLoss
        elif loss == 'NLL':
            losser = nn.NLLLoss
        self.loss = MaskedLoss(losser).to(self.device)
        self.infoNCELoss = InfoNCE(negative_mode='unpaired')

        # 定义优化器
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)

        # 打印模型参数大小
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        # Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        lr = self.optim.param_groups[0]['lr']
        losses, masks = [], []

        for num, data in enumerate(loader):
            self.optim.zero_grad()  # 梯度置为0
            # import ipdb;ipdb.set_trace()
            textf, visuf, acouf, qmask, umask, label = [d.to(self.device) for d in data[:-1]]

            lp_, x_a, x_l, x_v = self.model(torch.cat((textf,acouf,visuf), dim=-1), qmask)
            labels_ = label.view(-1) # batch*seq_len
            loss = self.loss(lp_, labels_, umask)
            # b = x_a.size(0)
            # x_a = x_a.reshape(b, -1)
            # x_l = x_l.reshape(b, -1)
            # x_v = x_v.reshape(b, -1)
            # l1 = self.infoNCELoss(x_a, x_a, x_l)
            # l2 = self.infoNCELoss(x_l, x_l, x_v)
            # l3 = self.infoNCELoss(x_v, x_v, x_a)
            # loss = loss + l1 + l2 + l3

            masks.append(umask.view(-1).cpu().numpy())
            losses.append(loss.item()*masks[-1].sum())
            loss.backward()
            self.optim.step()

        masks = np.concatenate(masks)
        avg_loss = round(np.sum(losses)/np.sum(masks), 4)

        return lr, avg_loss

    def eval_network(self, loader):
        self.eval()

        # 预测结果，实际结果
        preds, labels, masks = [], [], []

        with torch.no_grad():
            for num, data in enumerate(loader):
                textf, visuf, acouf, qmask, umask, label = [d.to(self.device) for d in data[:-1]]
                lp_, x_a, x_l, x_v =self.model(torch.cat((textf,acouf,visuf), dim=-1), qmask)
                # print(f'textf{textf.shape}  visuf{visuf.shape}  acouf{acouf.shape}') #100,512,100
                # print(f'input_size{(torch.cat((textf,acouf,visuf),dim=-1)).shape}')
                # log_prob, alpha, alpha_f, alpha_b = self.model(torch.cat((textf,acouf,visuf),dim=-1), qmask,umask,att2=True) # seq_len, batch, n_classes
                # print(f'log_prob{log_prob.shape}')
                # # log_prob, alpha, alpha_f, alpha_b = model(textf, qmask,umask,att2=True) # seq_len, batch, n_classes
                # # log_prob, alpha, alpha_f, alpha_b = self.model(textf,acouf,visuf, qmask, umask, att2=True)
                # lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
                # print(f'lp:{lp_.shape}')

                labels_ = label.view(-1) # batch*seq_len
                pred_ = torch.argmax(lp_,1) # batch*seq_len
                preds.append(pred_.data.cpu().numpy())
                labels.append(labels_.data.cpu().numpy())
                masks.append(umask.view(-1).cpu().numpy())
        
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
        df = pd.DataFrame({'trues': labels, 'preds': preds, 'masks': masks})
        df.to_csv("h.csv", sep=',', index=False)
        avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
        avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)

        return avg_accuracy, avg_fscore

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)