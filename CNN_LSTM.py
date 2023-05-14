import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.modules import dropout
class CNN_LSTM(nn.Module):
    def __init__(self, input_size=2):
        dim_hidden=64
        hidden = [256,128]
        super().__init__()
        self.conv=nn.Sequential(
            #------------------stage 1--------------------
            nn.Conv1d(input_size,dim_hidden,7,stride=1),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Conv1d(dim_hidden,dim_hidden*2,7,stride=1),
            nn.BatchNorm1d(dim_hidden*2),
            nn.LeakyReLU(),
            #
            nn.MaxPool1d(3,stride=1),
            nn.Dropout(0.1),
            )
        self.lstm1 =nn.LSTM(dim_hidden*2, hidden[0],
                            batch_first=True, bidirectional=False)
        self.lstm2=nn.LSTM( hidden[0], hidden[1],
                            batch_first=True, bidirectional=False)
        self.head=nn.Sequential(
            nn.Linear(hidden[1], 64),
            )
        self.dropout = nn.Dropout(0.1)
    #
    def attention_net(self, x, query, mask=None):
        # print('querr,,,,',x.shape)
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)

        p_attn = F.softmax(scores, dim = -1)
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn
    def forward(self, x):
        # print(x.shape)
        # # x = torch.unsqueeze(x, 1)
        # print(x.shape)
        x = x.squeeze(-1)
        x = x.permute(0,2,1)
        x=self.conv(x)
        # print(x.shape)
        #print(x.shape)
        x=x.permute(0,2,1)
        x,_=self.lstm1(x)
        x,_=self.lstm2(x)
        query = self.dropout(x)
        # print('x.shape')
        x, _ = self.attention_net(x, query)
        x = self.head(x)
        return x
if __name__=="__main__":
    x=torch.randn(64,32,2)
    net=CNN_LSTM()
    out=net(x)
    print(out.shape)