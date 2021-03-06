# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
# from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim

import sys
import os

import json
import time
import argparse
import numpy as np
from json import encoder
import logging

import os

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

from train import run_simple, RnnParameterData, generate_input_history, markov, \
    generate_input_long_history, generate_input_long_history2
from model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong, TrajPreAttnAvgLongUser2,TrajPreAttnAvgLongUser2_fine

def run(args):
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
                                  optim=args.optim, attn_type=args.attn_type,
                                  clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
                                  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path, model_method = args.model_method, pretrain = args.pretrain)
    logger.info('*' * 15 + 'loaded parameters' + '*' * 15)

    argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'voc_emb_size': args.voc_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,
            'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}
    logger.info('model_mode:{} history_mode:{} users:{}'.format(
        parameters.model_mode, parameters.history_mode, parameters.uid_size))

    if parameters.model_mode in ['simple', 'simple_long']:
        model = TrajPreSimple(parameters=parameters).cuda()
    elif parameters.model_mode == 'attn_avg_long_user':
        model = TrajPreAttnAvgLongUser2(parameters=parameters).cuda()
    elif parameters.model_mode == 'attn_local_long':
        model = TrajPreLocalAttnLong(parameters=parameters).cuda()


    if args.pretrain == 1:
        print("test")
        if 'taxi' in args.data_name:  # taxi
            model.load_state_dict(torch.load("")) 
        elif 'la' in args.data_name:  # la
            model.load_state_dict(torch.load(""))
        elif 'foursquare' in args.data_name:  # ny
            model.load_state_dict(torch.load("/home/bigdyl/jinsung/DeepMove2/ny_bestmodel/checkpoint/ep_32.m"))
        criterion = nn.NLLLoss().cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr, weight_decay=parameters.L2) 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step, factor=parameters.lr_decay, threshold=1e-3)
    else:
        print('train')
        criterion = nn.NLLLoss().cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr, weight_decay=parameters.L2) 
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step, factor=parameters.lr_decay, threshold=1e-3)
    
    if 'max' in parameters.model_mode:
        parameters.history_mode = 'max'
    elif 'avg' in parameters.model_mode:
        parameters.history_mode = 'avg'
    else:
        parameters.history_mode = 'whole'

    lr = parameters.lr
    metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'valid_acc': {}}

    candidate = parameters.data_neural.keys()  

    st = time.time()

    if 'long' in parameters.model_mode:
        long_history = True
    else:
        long_history = False

    if long_history is False:
        data_train, train_idx = generate_input_history(parameters.data_neural, 'train', mode2=parameters.history_mode,
                                                       candidate=candidate)
        data_test, test_idx = generate_input_history(parameters.data_neural, 'test', mode2=parameters.history_mode,
                                                     candidate=candidate)
    elif long_history is True:
        if parameters.model_mode == 'simple_long':
            data_train, train_idx = generate_input_long_history2(parameters.data_neural, 'train', candidate=candidate)
            data_test, test_idx = generate_input_long_history2(parameters.data_neural, 'test', candidate=candidate)
            data_valid, valid_idx = generate_input_long_history2(parameters.data_neural, 'valid', candidate=candidate)
        else:
            data_train, train_idx = generate_input_long_history(parameters.data_neural, 'train', candidate=candidate)
            data_test, test_idx = generate_input_long_history(parameters.data_neural, 'test', candidate=candidate)
            data_valid, valid_idx = generate_input_long_history(parameters.data_neural, 'valid', candidate=candidate)


    logger.info('*' * 15 + 'generated input: {}'.format(time.time() - st) + '*' * 15)
    SAVE_PATH = args.save_path
    tmp_path = 'checkpoint/'
    if not os.path.exists(SAVE_PATH + tmp_path):  # create checkpoint
        os.makedirs(SAVE_PATH + tmp_path)

    logger.info('*' * 15 + 'start training' + '*' * 15)
    for epoch in range(parameters.epoch):
        if args.pretrain == 2:
            break
        st = time.time()
        if args.pretrain == 0 or args.pretrain == 1:
            model, avg_loss = run_simple(data_train, train_idx, 'train', lr, parameters.clip, model, optimizer,
                                        criterion, parameters.model_mode)
            logger.info('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{} time:{}'.format(epoch, avg_loss, lr, time.time()-st))
            metrics['train_loss'].append(avg_loss)
            
        # validation
        avg_loss, avg_acc, users_acc = run_simple(data_valid, valid_idx, 'test', lr, parameters.clip, model, optimizer,
                                                criterion, parameters.model_mode,
                                                #   grid_eval=args.grid_eval,  # accuracy eval????????? grid mapping
                                                grid=parameters.grid_lookup)
        logger.info('==>Validation Acc:{:.4f} Loss:{:.4f}'.format(avg_acc, avg_loss))

        metrics['valid_loss'].append(avg_loss)
        metrics['accuracy'].append(avg_acc)
        metrics['valid_acc'][epoch] = users_acc

        save_name_tmp = 'ep_' + str(epoch) + '.m'
        torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)

        scheduler.step(avg_acc)
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:
            logger.info('lr_last > lr ... visit main.py line 152')
            load_epoch = np.argmax(metrics['accuracy'])
            load_name_tmp = 'ep_' + str(load_epoch) + '.m'
            model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
            logger.info('load epoch={} model state'.format(load_epoch))
        if epoch == 0:
            logger.info('single epoch time cost:{}'.format(time.time() - st))
        if lr <= 0.9 * 1e-5:
            break
        if args.pretrain == 2:
            break
    
    logger.info('*' * 15 + 'start testing' + '*' * 15)
    avg_loss, avg_acc, users_acc = run_simple(data_test, test_idx, 'test', lr, parameters.clip, model, optimizer,
                                        criterion, parameters.model_mode,look=parameters.vid_lookup)

    logger.info('==>Test Top1 Acc:{:.4f} Top5 Acc:{:.4f} Top10 Acc:{:.4f} MRR:{:.4f} Loss:{:.4f}'.format(
                avg_acc, 
                np.mean([users_acc[x][1] for x in users_acc]),
                np.mean([users_acc[x][2] for x in users_acc]),
                np.mean([users_acc[x][3] for x in users_acc]),  # overall mrr
                avg_loss))

    metrics['valid_loss'].append(avg_loss)
    metrics['accuracy'].append(avg_acc)
    metrics['valid_acc'][epoch] = users_acc

    save_name = 'res'
    json.dump({'args': argv, 'metrics': metrics}, fp=open(SAVE_PATH + save_name + '.rs', 'w'), indent=4)
    metrics_view = {'train_loss': [], 'valid_loss': [], 'accuracy': []}
    for key in metrics_view:
        metrics_view[key] = metrics[key]

    return avg_acc


def load_pretrained_model(config):
    if 'taxi' in config.data_name:  # taxi
        res = json.load(open("./taxi/ep6test/res.txt"))
    elif 'la' in config.data_name:  # la
        res = json.load(open("./la/final_model/res.txt"))
    elif 'foursquare' in config.data_name:  # ny
        res = json.load(open("/home/bigdyl/jinsung/DeepMove/codes/ny_test_2/res.rs"))
    args = Settings(config, res["args"])
    return args


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class Settings(object):
    def __init__(self, config, res):
        self.data_path = config.data_path
        self.save_path = config.save_path
        self.data_name = res["data_name"]
        self.epoch_max = res["epoch_max"]
        self.learning_rate = res["learning_rate"]
        self.lr_step = res["lr_step"]
        self.lr_decay = res["lr_decay"]
        self.clip = res["clip"]
        self.dropout_p = res["dropout_p"]
        self.rnn_type = res["rnn_type"]
        self.attn_type = res["attn_type"]
        self.L2 = res["L2"]
        self.history_mode = res["history_mode"]
        self.model_mode = res["model_mode"]
        self.optim = res["optim"]
        self.hidden_size = res["hidden_size"]
        self.tim_emb_size = res["tim_emb_size"]
        self.loc_emb_size = res["loc_emb_size"]
        self.uid_emb_size = res["uid_emb_size"]
        self.voc_emb_size = res["voc_emb_size"]
        self.model_method = config.model_method
        self.pretrain = 1


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int, default=500, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int, default=40, help="user id embeddings size")
    parser.add_argument('--voc_emb_size', type=int, default=50, help="words embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=10, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data_name', type=str, default='foursquare2')
    parser.add_argument('--learning_rate', type=float, default=5 * 1e-5)
    parser.add_argument('--lr_step', type=int, default=2)  # 3
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=30)
    parser.add_argument('--history_mode', type=str, default='avg', choices=['max', 'avg', 'whole'])
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--attn_type', type=str, default='dot', choices=['general', 'concat', 'dot'])
    parser.add_argument('--data_path', type=str, default='../data/Foursquare')
    parser.add_argument('--save_path', type=str, default='../out/')
    parser.add_argument('--model_mode', type=str, default='attn_avg_long_user')
    parser.add_argument('--load_checkpoint', type=int, default=None)  # checkpoint to load

    parser.add_argument('--model_method', type=int, default=0) # model_method = 0(G0E) 1(L2E) 2(G2E) 3(G5E)
    
    parser.add_argument('--jump', type=str, default='yes')
    parser.add_argument('--jump_method', type=str, default='gru')
    parser.add_argument('--jump_time', type=int, default=2)
    parser.add_argument('--pretrain', type=int, default=1)

    parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
    parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
    parser.add_argument('--hidden-dim', default=10, type=int, help='hidden dimensions to use')
    parser.add_argument('--rnn', default='lstm', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')   
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger = get_logger(logpath=os.path.join(args.save_path, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    ours_acc = run(args)
