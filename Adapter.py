import torch
import torch.nn as nn

class Adapter(nn.Module):       #这里定义Adapter的结构
    def __init__(self, dim):
        super(Adapter, self).__init__()

        # Adapter params
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        # 2D-CNN for spectrogram
        self.spec_down = nn.Linear(768, dim)
        self.spec_up = nn.Linear(dim, 768)
        nn.init.xavier_uniform_(self.spec_down.weight)
        nn.init.zeros_(self.spec_down.bias)
        nn.init.zeros_(self.spec_up.weight)
        nn.init.zeros_(self.spec_up.bias)
    def forward(self, x, f_dim, t_dim):
        x_down = self.spec_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.spec_up(x_down)

        return x_up


class AdapterBlock(nn.Module):      #这里使用的AdapterBlock，后续会替换VIT的Block
    def __init__(self, Encoder, dim,f_dim,t_dim):
        super(AdapterBlock, self).__init__()
        
        self.f_dim = f_dim
        self.t_dim = t_dim

        # Attention Layer
        self.norm1 = Encoder.norm1
        self.attn = Encoder.attn

        # Feed Forward Layers
        self.norm2 = Encoder.norm2
        self.mlp = Encoder.mlp

        # Conv Adapter
        self.conv1 = Adapter(dim=dim)
        self.conv2 = Adapter(dim=dim)

    def forward(self, x):
        # Attn skip connections
        x = x + self.attn(self.norm1(x)) + self.conv1(self.norm1(x), self.f_dim, self.t_dim)
        # MLP + skip conections
        x = x + self.mlp(self.norm2(x)) + self.conv2(self.norm2(x), self.f_dim, self.t_dim)
        return x