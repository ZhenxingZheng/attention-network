from opts import parser
from Dataset import VideoDataset
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
from Models import Spatial_TemporalNet
import torch.backends.cudnn as cudnn
import time
import os
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import scipy.io as scio
import math
import torch.nn.functional as F


def main():
    global args, best_prec1
    cudnn.benchmark = True
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    strat_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log = open(os.path.join(args.log_dir, strat_time + '.txt'), 'w')
    print (args.description)
    log.write(args.description + '\n')
    log.flush()
    print ('=======================Experimental Settings=======================\n')
    log.write('=======================Experimental Settings=======================\n')
    log.flush()
    print ('Using_Dataset:{0}  Batch_Size:{1}  Epoch:{2} '.format(args.dataset, args.batch_size, args.epoch))
    log.write('Using_Dataset:{0}  Batch_Size:{1}  Epoch:{2}'.format(args.dataset, args.batch_size, args.epoch) + '\n')
    log.flush()
    print ('Num_segments:{0}  Learning_rate:{1}  Attention_type:{2}\n'.format(args.segments, args.learning_rate, args.attention_type))
    log.write('Num_segments:{0}  Learning_rate:{1}  Attention_type:{2}\n'.format(args.segments, args.learning_rate, args.attention_type) + '\n')
    log.flush()
    print ('===================================================================\n')
    log.write('===================================================================\n')
    log.flush()

    train_loader = torch.utils.data.DataLoader(
        VideoDataset(root=args.root, list=args.train_video_list, num_segments=args.segments,
                     num_frames=args.frames,
                     transform=transforms.Compose([
                         transforms.Resize(256),
                         transforms.RandomCrop(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)


    test_loader = torch.utils.data.DataLoader(
        VideoDataset(root=args.root, list=args.test_video_list, num_segments=args.segments,
                     num_frames=args.frames, test_mode=True,
                     transform=transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    net = Spatial_TemporalNet(basemodel=args.base_model, dataset=args.dataset, segment=args.segments, attention_type=args.attention_type,
                                      hidden_size=args.hidden_size, img_dim=args.feature_size, kernel_size=args.kernel_size)
    net = torch.nn.DataParallel(net).cuda()

    for param in net.parameters():
        param.requires_grad = True

    def attentionloss(baseline, attention, target):
        target_temp = target.unsqueeze(1)
        baseline_temp = torch.gather(baseline, 1, target_temp).squeeze(0)
        attention_temp = torch.gather(attention, 1, target_temp).squeeze(0)
        selfloss = torch.max(torch.zeros(1).cuda(), baseline_temp - attention_temp + 0.1)

        return selfloss.mean()



    if args.cross:
        net.load_state_dict(torch.load('./model/2019-03-04 23:12:5914.pkl'))
        if args.target_dataset == 'hmdb':
            num_class = 51
        if args.target_dataset == 'ucf':
            num_class = 101
        setattr(net.module.temporal.reason_learned, 'fc', nn.Linear(1024, num_class).cuda())
        setattr(net.module.temporal.reason_auto, 'fc', nn.Linear(1024, num_class).cuda())
        setattr(net.module.temporal.reason_average, 'fc', nn.Linear(1024, num_class).cuda())
        print ('load pre-trained weights on Kinetics successfully ')

    if args.get_scores:
        net.load_state_dict(torch.load('./model/2019-03-11 08:45:40.pkl'))
        print ('begin to get class scores 2019-03-11 08:45:40')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    print ('doing experiments on ' + args.test_video_list)
    best_prec1 = 0
    for epoch in range(args.epoch):
        if not args.get_scores:
            adjust_learning_rate(optimizer, epoch, args)
            train(train_loader, net, criterion, optimizer, epoch, log, attentionloss)
            # torch.save(net.state_dict(), os.path.join(args.model_dir, strat_time + str(epoch) + '.pkl'))

            if (epoch + 1) % args.eval_freq == 0:
                prec1 = test(test_loader, net, epoch, log)
                if prec1 > best_prec1:
                    best_prec1 = prec1
                    torch.save(net.state_dict(), os.path.join(args.model_dir, strat_time + '.pkl'))
        else:
            print ('Begin Get Scores')
            gets(test_loader, net, epoch, args)
            print ('Done')
            break

def train(train_loader, net, criterion, optimizer, epoch, log, attentionloss):
    net.train()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    batch_time = AverageMeter()

    start = time.time()
    for step, (input, target) in enumerate(train_loader):
        input = Variable(input.view(-1, 3, 224, 224)).cuda()
        target = Variable(target).cuda(async=True)

        output_average, output_auto, output_learned, output = net(input)
        output_average = F.softmax(output_average, dim=1)
        output_auto = F.softmax(output_auto, dim=1)
        output_learned = F.softmax(output_learned, dim=1)

        loss = criterion(output, target) + attentionloss(output_average, output_auto, target) + attentionloss(output_average, output_learned, target)
        # loss = criterion(output,target)
        prec1, prec5 = compute_accuracy(output.data, target.data, topk=(1, 5))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() -start)
        start = time.time()

        if (step + 1) % args.print_freq == 0:
            NowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            output = ('Now Time {0}  Epoch:{1} || Step:{2}'
                      ' || Loss:{loss.avg:.4f}'
                      ' || Time:{batch_time.avg:.3f}'.format(NowTime, epoch, step + 1, loss=losses, batch_time=batch_time))
            print (output)
            log.write(output + '\n')
            log.flush()
    accuracy = ('Epoch:{0} || Prec@1: {top1.avg:.3f} || Prec@5: {top5.avg:.3f}').format(epoch + 1, top1=top1, top5=top5)
    print (accuracy)
    log.write(accuracy + '\n')
    log.flush()



def test(test_loader, net, epoch, log):
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for input, target in tqdm(test_loader):
        input = Variable(input.view(-1, 3, 224, 224)).cuda()
        target = Variable(target).cuda(async=True)

        output_average, output_auto, output_learned, output = net(input)
        output = torch.mean(output, dim=0, keepdim=True)
        prec1, prec5 = compute_accuracy(output.data, target.data, topk=(1, 5))
        top1.update(prec1)
        top5.update(prec5)

    NowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    accuracy = ('Testing Phrase ==>> Now Time {0} Epoch:{1} || Best Accuracy:{2} || Prec@1: {top1.avg:.3f} || Prec@5: {top5.avg:.3f}').format(NowTime, epoch, max(best_prec1, top1.avg), top1=top1, top5=top5, )
    print (accuracy)
    log.write(accuracy + '\n')
    log.flush()

    return top1.avg

def gets(test_loader, net, epoch, args):
    net.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if args.target_dataset == 'hmdb':
        mat = np.zeros((1, 51))
    elif args.target_dataset == 'ucf':
        mat = np.zeros((1, 101))
    elif args.target_dataset == 'kinetic':
        mat = np.zeros((1, 600))
    for step, (input, target) in enumerate(test_loader):
        print ('The Testing Number is {0}'.format(step))
        input = Variable(input.view(-1, 3, 224, 224)).cuda()
        target = Variable(target).cuda(async=True)

        output_average, output_auto, output_learned, output = net(input)
        output = torch.mean(output, dim=0, keepdim=True)
        prec1, prec5 = compute_accuracy(output.data, target.data, topk=(1, 5))
        mat = np.vstack((mat, output.cpu().data.view(1, -1).numpy()))

        top1.update(prec1)
        top5.update(prec5)


    NowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    accuracy = ('Testing Phrase ==>> Now Time {0} Epoch:{1} || Best Accuracy:{2} || Prec@1: {top1.avg:.3f} || Prec@5: {top5.avg:.3f}').format(NowTime, epoch, max(best_prec1, top1.avg), top1=top1, top5=top5, )
    print (accuracy)
    df = pd.DataFrame(mat[1:])
    df.to_excel(args.target_dataset + '.xlsx')


def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1)
    corrrect = pred.eq(target.view(-1, 1).expand_as(pred))

    store = []
    for k in topk:
        corrrect_k = corrrect[:,:k].float().sum()
        store.append(corrrect_k * 100.0 / batch_size)
    return store


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    if epoch in args.lr_step:
        args.learning_rate = args.learning_rate * 0.1
    lr = 0.5 * (1 + math.cos(epoch * math.pi / args.epoch)) * args.learning_rate

    # lr = lr * 0.1 ** (epoch // lr_step)
    print ('the learning rate is changed to {0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    main()
