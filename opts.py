import argparse

parser = argparse.ArgumentParser(description='Global and Local Knowledge-Aware Attention Network for Action Recognition')


#============================ Code Configs ============================
parser.add_argument('--train_video_list', default='list/hmdb_split1_train.txt', type=str) # ucf_trans_split1_train.txt
parser.add_argument('--test_video_list', default='list/hmdb_split1_test.txt', type=str)
parser.add_argument('--root', default='../Datasets/', type=str)
parser.add_argument('--dataset', default='kinetics', type=str)
parser.add_argument('--target_dataset', default='hmdb', type=str)
parser.add_argument('--log_dir', default='log', type=str)
parser.add_argument('--model_dir', default='model', type=str)
parser.add_argument('--get_scores', default=False, type=bool)
parser.add_argument('--description', type=str)
parser.add_argument('--pretrained', default=False, type=bool)
parser.add_argument('--cross', default='True', type=str)



#============================ Learning Configs ============================
parser.add_argument('--batch_size', default=30, type=int)
parser.add_argument('--workers', default=5, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=str)
parser.add_argument('--epoch', default=18, type=int)
parser.add_argument('--lr_step', default=[30, 40], type=list)
parser.add_argument('--print_freq', default=20, type=int)
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--momentum', default=0.9, type=float)


#============================ Model Configs ============================
parser.add_argument('--attention_type', default='all', type=str, help='average, auto and learned, all')
parser.add_argument('--feature_size', default=512, type=int)
parser.add_argument('--hidden_size', default=1024, type=int)
parser.add_argument('--segments', default=12, type=int)
parser.add_argument('--frames', default=1, type=int)
parser.add_argument('--base_model', default='resnet34', type=str)
parser.add_argument('--kernel_size', default=7, type=int)

