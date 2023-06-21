import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

import fcnet
from wideresnet import WideResNet
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def triplet_center_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    residual = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    positive = torch.diagonal(residual, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).to(device)*1e6
    negative = (residual + offset).min(-1)[0]
    loss = torch.clamp(positive + m - negative, min=0).mean()
    return loss

def get_f1_score(scores, labels, ratio):
    thresh = np.percentile(scores, ratio)
    y_pred = (scores >= thresh).astype(int)
    y_test = labels.astype(int)
    _, _, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    return f1_score

class TransformClassifier():
    def __init__(self, n_transform, args):
        self.n_transform = n_transform
        self.args = args
        self.WideResNet = WideResNet(self.args.depth, n_transform, self.args.widen_factor).to(device)
        self.optimizer = torch.optim.Adam(self.WideResNet.parameters())

    def fit_transform_classifier(self, x_train, x_test, y_test):
        print("Training")
        self.WideResNet.train()
        
        batch_size = self.args.batch_size
        N = x_train.shape[0]
        CE_loss = torch.nn.CrossEntropyLoss()
        ndf = 256
        lmbda_reg = 3*1e-6
        m = 0.1
        
        epochs = []
        scores = []
        for epoch in range(self.args.epochs):
            random_samples = np.random.permutation(N//self.n_transform)
            random_samples = np.concatenate([np.arange(self.n_transform) + random_samples[i]*self.n_transform for i in range(len(random_samples))])
            assert len(random_samples) == N
            zs_train = torch.zeros((len(x_train), ndf)).to(device)
            diffs_all = []

            for i in range(0, len(x_train), batch_size):
                batch_range = min(batch_size, len(x_train) - i)
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(x_train[random_samples[idx]]).float().to(device)
                zs_tc, zs_ce = self.WideResNet(xs)

                zs_train[idx] = zs_tc
                y_train = torch.from_numpy(np.tile(np.arange(self.n_transform), batch_range//self.n_transform)).long().to(device)
                zs = torch.reshape(zs_tc, (batch_range//self.n_transform, self.n_transform, ndf))

                means = zs.mean(0).unsqueeze(0)
                diffs = -((zs.unsqueeze(2).detach().cpu().numpy() - means.unsqueeze(1).detach().cpu().numpy()) ** 2).sum(-1)
                diffs_all.append(torch.diagonal(torch.tensor(diffs), dim1=1, dim2=2))

                if self.args.analytic_margin == 1:
                    lasso = 0
                    for param in self.WideResNet.parameters():
                        lasso += torch.abs(param).sum()
                    
                    lasso *= lmbda_reg
                    m = lasso

                tc = triplet_center_loss(zs, m)
                ce = CE_loss(zs_ce, y_train)
                
                if self.args.reg:
                    loss = ce + self.args.lmbda * tc + 10 *(zs*zs).mean()
                else:
                    loss = ce + self.args.lmbda * tc

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.WideResNet.eval()
            zs_train = torch.reshape(zs_train, (N//self.n_transform, self.n_transform, ndf))
            means = zs_train.mean(0, keepdim=True)
            
            with torch.no_grad():
                batch_size = batch_size
                y_pred = np.zeros((len(y_test), self.n_transform))
                for i in range(0, len(x_test), batch_size):
                    batch_range = min(batch_size, len(x_test) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_test[idx]).float().to(device)

                    zs, _ = self.WideResNet(xs)
                    zs = torch.reshape(zs, (batch_range // self.n_transform, self.n_transform, ndf))

                    diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)
                    diffs_eps = self.args.eps * torch.ones_like(diffs)
                    diffs = torch.max(diffs, diffs_eps)
                    pred_probs = torch.nn.functional.log_softmax(-diffs, dim=2)

                    zs_reidx = np.arange(batch_range // self.n_transform) + i // self.n_transform
                    y_pred[zs_reidx] = -torch.diagonal(pred_probs, 0, 1, 2).cpu().data.numpy()

                y_pred = y_pred.sum(1)
                print("Epoch:", epoch, ", AUC: ", roc_auc_score(y_test, -y_pred))
                epochs.append(epoch)
                scores.append(roc_auc_score(y_test, -y_pred))
        
        filename = str(self.args.class_index) + "_epoch_and_scores.csv"
        df = pd.DataFrame({"epochs": epochs, "scores": scores})
        df.to_csv("./"+filename)

class TransformTabularClassifier():
    def __init__(self, args):
        self.dataset = args.dataset
        self.args = args
        self.n_rotation = 256
        self.d_out = 32
        self.ndf = 8

        self.net = fcnet.FCNET(self.d_out, self.ndf, self.n_rotation).to(device)
        fcnet.weights_init(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def fit_transform_classifier(self, x_train, x_test, y_test, ratio):
        labels = torch.arange(self.n_rotation).unsqueeze(0).expand((self.args.batch_size, self.n_rotation)).long().to(device)
        CE_loss = nn.CrossEntropyLoss()
        m = 1
        lmbda_reg = 6*1e-6
        # print('Training')
        
        for epoch in range(self.args.epochs):
            self.net.train()
            random_samples = np.random.permutation(len(x_train)) #1839
            n_batch = 0
            sum_zs = torch.zeros((self.ndf, self.n_rotation)).to(device) #(8, 256)

            for i in range(0, len(x_train), self.args.batch_size):
                self.net.zero_grad()
                batch_range = min(self.args.batch_size, len(x_train) - i)
                y_train = labels
                if batch_range == len(x_train) - i:
                    y_train = torch.arange(self.n_rotation).unsqueeze(0).expand((len(x_train) - i, self.n_rotation)).long().to(device)
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(x_train[random_samples[idx]]).float().to(device)
                zs_tc, zs_ce = self.net(xs)
                sum_zs += zs_tc.mean(0)
                zs_tc = zs_tc.permute(0, 2, 1)

                if self.args.analytic_margin == 1:
                    lasso = 0
                    for param in self.net.parameters():
                        lasso += torch.abs(param).sum()
                    
                    lasso *= lmbda_reg
                    m = lasso
                
                ce = CE_loss(zs_ce, y_train)
                loss = self.args.lmbda * triplet_center_loss(zs_tc, m) + ce
                loss.backward()
                self.optimizer.step()
                n_batch += 1

            means = sum_zs.t() / n_batch
            means = means.unsqueeze(0)
            self.net.eval()

            with torch.no_grad():
                y_pred = np.zeros((len(y_test), self.n_rotation))
                for i in range(0, len(x_test), self.args.batch_size):
                    batch_range = min(self.args.batch_size, len(x_test) - i)
                    idx = np.arange(batch_range) + i
                    xs = torch.from_numpy(x_test[idx]).float().to(device)
                    zs, _ = self.net(xs)
                    zs = zs.permute(0, 2, 1)
                    
                    diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)
                    diffs_eps = self.args.eps * torch.ones_like(diffs)
                    diffs = torch.max(diffs, diffs_eps)
                    
                    pred_probs = torch.nn.functional.log_softmax(-diffs, dim=2)

                    y_pred[idx] = -torch.diagonal(pred_probs, 0, 1, 2).cpu().data.numpy()

                y_pred = y_pred.sum(1)
                f1_score = get_f1_score(y_pred, y_test, ratio)
                # print("Epoch:", epoch, ", fscore: ", f1_score)
                
        return f1_score