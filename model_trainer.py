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
from models.lstm import BiLSTM
from models.lsthm_newz import MARN1_newz
from models.lsthm_azs import MARN1_azs
from models.lsthm_mf import MARN1_mf
from models.lsthm_la import MARN1_la
from models.lsthm_cf import MARN1_cf
from models.lsthm_sp import MARN1_sp
from models.lsthm_sps import MARN1_sps
from models.lsthm_nsps import MARN1_nsps
from models.lsthm_onlysp import MARN1_onlysp
from models.lsthm_no_en import MARN1_no_en


class ModelTrainer(nn.Module):

    def __init__(self, device, lr, test_step, lr_decay, model, loss, n_classes, dataset, **kwargs):
        super(ModelTrainer, self).__init__()
        self.device = device
        self.dataset = dataset
        # 定义模型
        if model == 'DialogueRNN':
            D_m = 712
            D_g = 500
            D_p = 500
            D_e = 300
            D_h = 300

            self.model = BiModel(D_m, D_g, D_p, D_e, D_h,
                                n_classes=6,
                                listener_state=True,
                                context_attention='general',
                                dropout_rec=0.1,
                                dropout=0.1).to(self.device)

        elif model=='MARN':
            self.model=MARN().to(self.device)
        elif model == 'BiLSTM':
            self.model=BiLSTM().to(self.device)
        elif model == 'MARN1_newz':
            self.model=MARN1_newz().to(self.device)
        elif model == 'MARN1_azs':
            self.model=MARN1_azs(n_classes).to(self.device)
        elif model == 'MARN1_mf':
            self.model=MARN1_mf(n_classes).to(self.device)
        elif model == 'MARN1_la':
            self.model=MARN1_la(n_classes).to(self.device)
        elif model == 'MARN1_cf':
            self.model=MARN1_cf(n_classes).to(self.device)
        elif model == 'MARN1_sp':
            self.model=MARN1_sp(n_classes).to(self.device)
        elif model == 'MARN1_sps':
            self.model=MARN1_sps(n_classes).to(self.device)
        elif model == 'MARN1_nsps':
            self.model=MARN1_nsps(n_classes, dataset).to(self.device)
        elif model == 'MARN1_onlysp':
            self.model=MARN1_onlysp(n_classes).to(self.device)
        elif model == 'MARN1_no_en':
            self.model=MARN1_no_en(n_classes, dataset).to(self.device)
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
            if self.dataset == 'IEMOCAP':
                r1, r2, r3, r4, visuf, acouf, qmask, umask, label = [d.to(self.device) for d in data[:-1]]
            else:
                r1, r2, r3, r4, textf, acouf, qmask, umask, label = [d.to(self.device) for d in data[:-1]]
            
            textf = (r1 + r2 + r3 + r4) / 4
            lp_, x_a, x_l  =self.model(torch.cat((textf,acouf), dim=-1), qmask, umask)

            labels_ = label.view(-1) # batch*seq_len
            
            loss = self.loss(lp_, labels_, umask)

            # b = x_a.size(1)
            # x_a = x_a.permute(1, 0, 2).reshape(b, -1)
            # x_l = x_l.permute(1, 0, 2).reshape(b, -1)
            # l1 = self.infoNCELoss(x_a, x_a, x_l)
            # loss = loss + l1

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
                if self.dataset == 'IEMOCAP':
                    r1, r2, r3, r4, visuf, acouf, qmask, umask, label = [d.to(self.device) for d in data[:-1]]
                else:
                    r1, r2, r3, r4, textf, acouf, qmask, umask, label = [d.to(self.device) for d in data[:-1]]
                
                textf = (r1 + r2 + r3 + r4) / 4
                lp_, x_a, x_l =self.model(torch.cat((textf,acouf), dim=-1), qmask, umask)

                labels_ = label.view(-1) # batch*seq_len
                pred_ = torch.argmax(lp_,1) # batch*seq_len
                preds.append(pred_.data.cpu().numpy())
                labels.append(labels_.data.cpu().numpy())
                masks.append(umask.view(-1).cpu().numpy())
        
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)

        df = pd.DataFrame({'preds': preds, 'labels': labels, 'masks': masks})
        df.to_csv('res.csv', index=False)
        print('finish')

        avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
        avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)

        # w = self.state_dict()['model.w'].item() 
        # v = self.state_dict()['model.v'].item() 
        # w1 = self.state_dict()['model.w1'].item() 
        # v1 = self.state_dict()['model.v1'].item() 
        # p = self.state_dict()['model.p'].tolist() 

        w, v, w1, v1 = [], [], [], []
        # {'text': p[0], 'audio': p[1], 'l2a': p[2], 'a2l': p[3], 'sp': p[4]}
        return avg_accuracy, avg_fscore, {}

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