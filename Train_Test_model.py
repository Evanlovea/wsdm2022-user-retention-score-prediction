#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import os
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from CNN_LSTM import CNN_LSTM
import warnings
import sys
import copy
sys.path.append("../")
warnings.filterwarnings('ignore')
data_dir = "./data/wsdm_train_data2/"
SAVE_PATH = './WSDM_AQI_best_model0119_2.pth'
output_dir = "./data/"
test_data_dir = os.path.join(data_dir, 'test_data.txt')
train_data_dir = os.path.join(data_dir, 'train_data.txt')
output_filename = os.path.join(output_dir, "87.86submission_220120_4.csv")
### random seed
def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# random_seed(42)
# get my device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# hyper_params
learning_rate = 0.001  # 学习率
batch_size = 64
num_workers = 0
epoch = 100
early_stop_epoch = 15
# train data
data = pd.read_csv(train_data_dir, sep="\t")
data["launch_seq"] = data.launch_seq.apply(lambda x: json.loads(x))
data["duration_prefer"] = data.duration_prefer.apply(lambda x: json.loads(x))
data["interact_prefer"] = data.interact_prefer.apply(lambda x: json.loads(x))
# ----------------------
data["playtime_seq"] = data.playtime_seq.apply(lambda x: json.loads(x))
# shuffle data
data = data.sample(frac=1).reset_index(drop=True)
data = data.drop(columns=['device_rom','device_ram'])
# eval(data['playtime_seq'].iloc[10])
# data.iloc[1].values
# print(min(len(data['launch_seq'].iloc[i]) for i in range (data.shape[0])))
# print(min(len(data['playtime_seq'].iloc[i]) for i in range (data.shape[0])))
# print(min(len(data['duration_prefer'].iloc[i]) for i in range (data.shape[0])))
# print(min(len(data['interact_prefer'].iloc[i]) for i in range (data.shape[0])))
# print(max(len(data['launch_seq'].iloc[i]) for i in range (data.shape[0])))
# print(max(len(data['playtime_seq'].iloc[i]) for i in range (data.shape[0])))
# print(max(len(data['duration_prefer'].iloc[i]) for i in range (data.shape[0])))
# print(max(len(data['interact_prefer'].iloc[i]) for i in range (data.shape[0])))
# print(max(len(data['launch_seq'].iloc[i]) for i in range (data.shape[0])))
# print(data['label'])
# data['launch_seq'].count

# In[29]:
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
#### 使用LDS(label distribution smooth)
# define the loss function
# 这里我们仅仅使用weighted_l1_loss
def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_mse_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_l1_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_huber_loss(inputs, targets, beta=1., weights=None):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def get_lds_kernel_window(kernel, ks, sigma):
    """

    Args:
        kernel:
        ks: kernel size
        sigma: standard deviation

    Returns:

    """
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'triang':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


# Option 4: To train the Label Distribution Smoothing (LDS) model:
# LDS params
reweight = 'sqrt_inv'
lds = True
lds_kernel = 'laplace'
lds_ks = 5 # 5
lds_sigma = 0.01 # 2


class MyDataset(Dataset):
    def __init__(self, df, reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        super(MyDataset, self).__init__()
        self.df = df

        self.feat_col = list(set(self.df.columns)-set(['user_id','end_date','label','launch_seq', 'playtime_seq', 'duration_prefer', 'interact_prefer']))
        self.df_feat = self.df[self.feat_col]
        # 定义LDS weights
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __getitem__(self, index):
        launch_seq = self.df['launch_seq'].iloc[index]
        playtime_seq = self.df["playtime_seq"].iloc[index]
        duration_prefer = self.df["duration_prefer"].iloc[index]
        interact_prefer = self.df['interact_prefer'].iloc[index]
        # play_ratio_seq = self.df['play_ratio_seq'].iloc[index]
        feat = self.df_feat.iloc[index].values.astype(np.float32)
        label = self.df['label'].iloc[index]
        # weights
        weight = np.asarray([self.weights[index]]).astype(np.float32) if self.weights is not None else np.asarray([np.float32(1.)])
        if(isinstance(playtime_seq, str)):
            playtime_seq = eval(playtime_seq)
        else:
            pass
        weight = torch.FloatTensor(weight)
        launch_seq = torch.FloatTensor(launch_seq)
        playtime_seq = torch.FloatTensor(playtime_seq)
        duration_prefer = torch.FloatTensor(duration_prefer)
        interact_prefer = torch.FloatTensor(interact_prefer)
        feat = torch.FloatTensor(feat)
        # play_ratio_seq = torch.FloatTensor(play_ratio_seq)
        label = torch.FloatTensor([label])
        return launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label, weight

    def __len__(self):
        return len(self.df)

    def _prepare_weights(self, reweight=None, max_target=7, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['label'].tolist()
        # mbr
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights


train_data = MyDataset(data.iloc[:-6000], reweight=reweight,
                 lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)
val_data = MyDataset(data.iloc[-6000:])
train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=False, num_workers=num_workers)

# process test data set
test_data = pd.read_csv(test_data_dir, sep="\t")
test_data["launch_seq"] = test_data.launch_seq.apply(lambda x:json.loads(x))
test_data["playtime_seq"] = test_data.playtime_seq.apply(lambda x: json.loads(x))
test_data["duration_prefer"] = test_data.duration_prefer.apply(lambda x: json.loads(x))
test_data["interact_prefer"] = test_data.interact_prefer.apply(lambda x: json.loads(x))
# test_data['play_ratio_seq'] = test_data.play_ratio_seq.apply(lambda x: json.loads(x))
test_data = test_data.drop(columns=['device_rom', 'device_ram'])
test_data['label'] = 0
test_dataset = MyDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)


# define my_model
class MyModel(nn.Module):
    # 前7天，前14
    def __init__(self):
        super(MyModel, self).__init__()

        self.seq_lstm = CNN_LSTM()
        # self.playtime_seq_gru = CNN_Bi()
        self.fc1 = nn.Linear(100, 64)
        self.ac1 = nn.SELU(inplace=True)
        self.fc2 = nn.Linear(64, 32)
        self.ac2 = nn.SELU(inplace=True)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, launch_seq, playtime_seq, duration_prefer, interact_prefer, feat):
        launch_seq = launch_seq.reshape((-1, 32, 1))
        playtime_seq = playtime_seq.reshape((-1, 32, 1))
        launch_seq = launch_seq.unsqueeze(2)
        playtime_seq = playtime_seq.unsqueeze(2)
        seq = torch.cat((launch_seq, playtime_seq), 2)  # (batch_size, sequence_len, channel_size)
        seq_feat = self.seq_lstm(seq)
        all_feat = torch.cat((seq_feat, duration_prefer, interact_prefer, feat), 1)
        all_feat_fc1 = self.fc1(all_feat)
        feat_fc_relu = self.ac1(all_feat_fc1)
        all_feat_fc2 = self.fc2(feat_fc_relu)
        feat_fc_relu2 = self.ac2(all_feat_fc2)
        all_feat_fc3 = self.fc3(feat_fc_relu2)
        return all_feat_fc3


model = MyModel().to(device)

# model training
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()
# criterion = nn.SmoothL1Loss()
def cal_score(pred, label):
    pred = np.array(pred)
    label = np.array(label)

    diff = (pred - label) / 7
    diff = np.abs(diff)

    score = 100 * (1 - np.mean(diff))
    return score


# In[36]:


# 学习率衰减带重启的随机梯度下降算法（SGDR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)


def train(model, train_loader, optimizer,device):
    model.train()
    train_loss = []
    pred_list = []
    label_list=[]
    # print("train_loader", train_loader)
    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label, weights in tqdm(train_loader):
        # to device
        launch_seq = launch_seq.to(device)
        playtime_seq = playtime_seq.to(device)
        duration_prefer = duration_prefer.to(device)
        interact_prefer = interact_prefer.to(device)
        # play_ratio_seq = play_ratio_seq.to(device)
        feat = feat.to(device)
        label = label.to(device)
        weights = weights.to(device)
        # 梯度清零
        
        optimizer.zero_grad()
        # 正向传播求损失
        pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat)
        loss = weighted_l1_loss(pred, targets=label, weights=weights)
        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"
        # print(pred)
        # calculate score
        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())
        # 反向传播求梯度
#         loss = criterion(pred, label)

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss.append(loss.item())
    score = cal_score(pred_list,label_list)
    return np.mean(train_loss), score


# In[37]:


def validate(model, val_loader, optimizer,device):
    # 在模型中，我们通常会加上Dropout层和batch normalization层，
    # 在模型预测阶段，我们需要将这些层设置到预测模式，model.eval()就是帮我们一键搞定的
    model.eval()
    val_loss = []
    pred_list = []
    label_list = []
    for launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label, weights in tqdm(val_loader):
        launch_seq = launch_seq.to(device)
        playtime_seq = playtime_seq.to(device)
        duration_prefer = duration_prefer.to(device)
        interact_prefer = interact_prefer.to(device)
        # play_ratio_seq = play_ratio_seq.to(device)
        feat =feat.to(device)
        label = label.to(device)
        weights = weights.to(device)
        optimizer.zero_grad()

        pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat)
        # loss = criterion(pred, label)
        loss = weighted_l1_loss(pred, targets=label, weights=weights)

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())
        val_loss.append(loss.item())
    score = cal_score(pred_list, label_list)
    return np.mean(val_loss), score


def predict(model, test_loader):
    model.eval()
    test_pred = []
    for (launch_seq, playtime_seq, duration_prefer, interact_prefer, feat, label, _) in tqdm(test_loader):
        launch_seq = launch_seq.to(device)
        playtime_seq = playtime_seq.to(device)
        duration_prefer = duration_prefer.to(device)
        interact_prefer = interact_prefer.to(device)
        # play_ratio_seq = play_ratio_seq.to(device)
        feat = feat.to(device)
        # label = label.to(device)
        pred = model(launch_seq, playtime_seq, duration_prefer, interact_prefer, feat)
        test_pred.append(pred.cpu().detach().numpy())
    return test_pred

# In[39]:

best_val_score = float('-inf')
last_improve=0
best_model=None
for epoch in range(epoch):
    train_loss, train_score = train(model, train_loader, optimizer, device)
    val_loss, val_score = validate(model, val_loader, optimizer,device)
    # 加入早停机制
    if val_score > best_val_score:
        best_val_score = val_score
        best_model = copy.deepcopy(model)
        last_improve = epoch
        improve = '*'
    else:
        improve = ''
    if epoch - last_improve > early_stop_epoch:
        print("------提前结束！------")
        break
    print(
            f'Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {val_loss}, Train Score: {train_score}, Valid Score: {val_score} {improve}{improve}'
        )

print("Offline_Best_Val_Score:{}".format(best_val_score))
model = best_model
# save my best model, only save the model parameters
torch.save(model.state_dict(), SAVE_PATH)
# # load my model
# the_model = MyModel()
# the_model.load_state_dict(torch.load(SAVE_PATH))

# predict the test data set
test_pred = predict(model, test_loader)
test_pred = np.vstack(test_pred)
test_data["prediction"] = test_pred[:, 0]
test_data = test_data[["user_id", "prediction"]]
# can clip outputs to [0, 7] or use other tricks
test_data.to_csv(output_filename, index=False, header=False, float_format="%.2f")
print("输出文件成功！")

