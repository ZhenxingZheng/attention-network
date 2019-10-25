import torch.nn as nn
from torch.autograd import Variable
import torch
import torchvision.models as models
import torch.nn.functional as F
from sru import SRU, SRUCell
import Attention


class Recurrent_model(nn.Module):
    def __init__(self, img_dim=512, num_segments=12, hidden_size=1024, num_class=51):
        super(Recurrent_model, self).__init__()
        self.img_dim = img_dim
        self.num_segments = num_segments
        self.num_class = num_class

        self.rnn = SRU(img_dim, hidden_size,
                       num_layers = 3,
                       dropout = 0.5,
                       bidirectional = False,
                       layer_norm=False,
                       highway_bias=0,
                       rescale=True
        )

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hidden_size, self.num_class)

    def forward(self, x):
        r_out, h_state = self.rnn(x.transpose(0, 1))
        out_step = self.fc(self.dropout(r_out))
        output = torch.mean(out_step, dim=0)

        return output


class Temporal_Net(nn.Module):
    def __init__(self, dataset='hmdb', segments=12, attention='average',  hidden_size=512, img_dim=512, kernel_size=7):
        super(Temporal_Net, self).__init__()
        self.dataset = dataset
        self.sequence = segments
        self.attention_type = attention
        self.hidden_size = hidden_size
        self.img_dim = img_dim
        self.kernel_size = kernel_size

        if self.dataset == 'hmdb':
            self.num_class = 51
        elif self.dataset == 'ucf':
            self.num_class = 101
        elif self.dataset == 'kinetics':
            self.num_class = 600

        if self.attention_type == 'all':
            print ('using all attention  for action recognition')
            self.attention_average = Attention.Attention_average(sequence=self.sequence, img_dim=self.img_dim, kernel_size=self.kernel_size)
            self.attention_auto = Attention.Attentnion_auto(sequence=self.sequence, img_dim=self.img_dim, kernel_size=self.kernel_size)
            self.attention_learned = Attention.Attention_learned(sequence=self.sequence, img_dim=self.img_dim, kernel_size=self.kernel_size)
        self.reason_average = Recurrent_model(img_dim=self.img_dim, hidden_size=self.hidden_size, num_class=self.num_class, num_segments=self.sequence)
        self.reason_auto = Recurrent_model(img_dim=self.img_dim * 2, hidden_size=self.hidden_size, num_class=self.num_class, num_segments=self.sequence)
        self.reason_learned = Recurrent_model(img_dim=self.img_dim * 2, hidden_size=self.hidden_size, num_class=self.num_class, num_segments=self.sequence)


    def forward(self, frame_features):

        ready_recurrent_average = self.attention_average(frame_features) # 2 * 12 * 512
        output_average = self.reason_average(ready_recurrent_average)

        ready_recurrent_auto = self.attention_auto(frame_features)  # 2 * 12 * 512
        recurrent_auto = torch.cat([ready_recurrent_average, ready_recurrent_auto], dim=2)
        output_auto = self.reason_auto(recurrent_auto)


        ready_recurrent_learned = self.attention_learned(frame_features)  # 2 * 12 * 512
        recurrent_learned = torch.cat([ready_recurrent_average, ready_recurrent_learned], dim=2)
        output_learned = self.reason_learned(recurrent_learned)

        output = (output_average + output_auto + output_learned) / 3

        return [output_average, output_auto, output_learned, output]


class Spatial_Net(nn.Module):
    def __init__(self,  basemodel='resnet34'):
        super(Spatial_Net, self).__init__()
        self.basemodel = basemodel
        self.prepare_basemodel(self.basemodel)


    def forward(self, frame):
        output = self.net(frame)
        return output

    def prepare_basemodel(self, basemodel):
        basemodel = getattr(models, basemodel)(True)
        module_list = list(basemodel.children())
        del module_list[-1]
        del module_list[-1]

        self.net = nn.Sequential(*module_list)


class Spatial_TemporalNet(nn.Module):
    def __init__(self, basemodel='resnet34', dataset='kinetics', segment=12, attention_type='all',
                 hidden_size=1024, img_dim=512, kernel_size=7):
        super(Spatial_TemporalNet, self).__init__()
        self.spatial = Spatial_Net(basemodel=basemodel)
        self.temporal = Temporal_Net(dataset=dataset, segments=segment, attention=attention_type,  hidden_size=hidden_size, img_dim=img_dim, kernel_size=kernel_size)

    def forward(self, x):
        output_spatial = self.spatial(x) # 24 * 512 * 7 * 7
        output_temporal = self.temporal(output_spatial)
        return output_temporal


if __name__ == '__main__':
    print('haha')
    faka_data = Variable(torch.randn(2, 12, 3, 224, 224)).cuda().view(-1, 3, 224, 224)
    net = Spatial_TemporalNet().cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load('./model/2019-03-04 23:12:5910.pkl'))

    output = net(faka_data)
    print (output[0].size())

