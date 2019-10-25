import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Attention_average(nn.Module):
    def __init__(self, sequence, img_dim, kernel_size):
        super(Attention_average, self).__init__()
        self.sequence = sequence
        self.img_dim = img_dim
        self.kernel_size = kernel_size

    def forward(self, x):
        output = self.pooling(x).view(-1, self.sequence, self.img_dim)
        return output

    def pooling(self, x):
        output = torch.mean(torch.mean(x, dim=3), dim=2)
        return output


class Attentnion_auto(nn.Module):
    def __init__(self, sequence, img_dim, kernel_size,):
        super(Attentnion_auto, self).__init__()
        self.sequence = sequence
        self.img_dim = img_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        feature_pow = torch.pow(x, 2)
        feature_map = torch.mean(feature_pow, dim=1).view(-1, 1, self.kernel_size, self.kernel_size)
        feature_map = self.conv(feature_map).view(-1, self.kernel_size ** 2)
        feature_weight = F.softmax(feature_map, dim=-1).view(-1, 1, self.kernel_size, self.kernel_size).expand_as(x)
        out_map = feature_weight * x
        output = torch.sum(torch.sum(out_map, dim=3), dim=2)

        return output.view(-1, self.sequence, self.img_dim)



class Attention_learned(nn.Module):
    def __init__(self, sequence, img_dim, kernel_size, bottle_neck=128):
        super(Attention_learned, self).__init__()
        self.kernel_size = kernel_size
        self.im_dim = img_dim
        self.sequence = sequence
        # self.alpha = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        self.linear = nn.Sequential(
            nn.Linear(self.im_dim, bottle_neck),
            nn.Tanh(),
            nn.Linear(bottle_neck, 1),
            nn.Tanh(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(self.kernel_size ** 2, self.kernel_size ** 2, 1),
            # nn.Sigmoid(),
        )


    def forward(self, outhigh):
        outhigh = outhigh.view(-1, self.im_dim, self.kernel_size * self.kernel_size).transpose(1, 2)
        weight = self.linear(outhigh).squeeze(-1)
        attention = F.softmax(weight, dim=-1).unsqueeze(-1)
        attention_data = outhigh * attention
        descriptor = torch.sum(attention_data, dim=1)

        return descriptor.view(-1, self.sequence, self.im_dim)



if __name__ == '__main__':
    fake_data = Variable(torch.randn(24, 512, 7, 7))
    net = Attentnion_auto(sequence=12, img_dim=512, kernel_size=7)
    print (net(fake_data).size())
