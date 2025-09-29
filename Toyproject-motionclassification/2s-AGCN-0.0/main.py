#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.autograd import Variable  # ### OLD: Variable 사용
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
import re  # [CHANGED] 가중치 파일명에서 step 파싱용

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args.get('debug', False):
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0

        # [CHANGED] test 단계에서 optimizer 생성/호출 안 함
        # ### OLD:
        # self.load_model()
        # self.load_optimizer()
        # self.load_data()
        self.load_model()
        self.load_data()
        if getattr(self.arg, 'phase', 'train') == 'train':
            self.load_optimizer()

        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        # [CHANGED] CPU/단일/멀티GPU 안전한 디바이스 선택
        # ### OLD:
        # output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        # self.output_device = output_device
        dev_ids = self.arg.device if isinstance(self.arg.device, (list, tuple)) else [self.arg.device]
        dev_ids = list(dev_ids)
        use_cuda = torch.cuda.is_available() and len(dev_ids) > 0 and (dev_ids[0] is not None) and (int(dev_ids[0]) >= 0)
        output_device = int(dev_ids[0]) if use_cuda else 'cpu'
        self.output_device = output_device

        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)

        # [CHANGED] CPU 지원
        # ### OLD:
        # self.model = Model(**self.arg.model_args).cuda(output_device)
        # self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.model = Model(**self.arg.model_args)
        if use_cuda:
            self.model = self.model.cuda(output_device)
            self.loss = nn.CrossEntropyLoss().cuda(output_device)
        else:
            self.model = self.model.to('cpu')
            self.loss = nn.CrossEntropyLoss().to('cpu')
        print(self.model)

        # [CHANGED] 가중치 로드: step 파싱/PKL 바이너리/strict 로드 개선
        if self.arg.weights:
            # ### OLD:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            base = os.path.basename(self.arg.weights)
            m = re.search(r'-(\d+)(?:\.converted)?\.pt$', base)
            self.global_step = int(m.group(1)) if m else 0

            self.print_log('Load weights from {}.'.format(self.arg.weights))
            # ### OLD:
            # if '.pkl' in self.arg.weights:
            #     with open(self.arg.weights, 'r') as f:
            #         weights = pickle.load(f)
            # else:
            #     weights = torch.load(self.arg.weights)
            if self.arg.weights.endswith('.pkl'):
                with open(self.arg.weights, 'rb') as f:  # 바이너리 모드
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights, map_location='cpu')

            # [CHANGED] DataParallel 'module.' 제거 + 디바이스 이동은 나중에
            # ### OLD:
            # weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            if isinstance(weights, dict) and all(isinstance(k, str) for k in weights.keys()):
                weights = OrderedDict([[k.split('module.')[-1], v] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if isinstance(weights, dict) and weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            # [CHANGED] strict=True 우선, 실패시 partial 로드
            # ### OLD:
            # try:
            #     self.model.load_state_dict(weights)
            # except:
            #     state = self.model.state_dict()
            #     diff = list(set(state.keys()).difference(set(weights.keys())))
            #     print('Can not find these weights:')
            #     for d in diff:
            #         print('  ' + d)
            #     state.update(weights)
            #     self.model.load_state_dict(state)
            try:
                self.model.load_state_dict(weights, strict=True)
            except:
                state = self.model.state_dict()
                diff = [k for k in state.keys() if k not in weights]
                if diff:
                    print('Can not find these weights:')
                    for d in diff:
                        print('  ' + d)
                state.update({k: v for k, v in weights.items() if k in state})
                self.model.load_state_dict(state, strict=False)

        # [CHANGED] 멀티GPU는 실제 존재할 때만
        # ### OLD:
        # if type(self.arg.device) is list:
        #     if len(self.arg.device) > 1:
        #         self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=output_device)
        if use_cuda and len(dev_ids) > 1:
            self.model = nn.DataParallel(
                self.model,
                device_ids=[int(i) for i in dev_ids],
                output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        # [CHANGED] ReduceLROnPlateau에서 verbose 제거(버전 호환)
        # ### OLD:
        # self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
        #                                       patience=10, verbose=True,
        #                                       threshold=1e-4, threshold_mode='rel',
        #                                       cooldown=0)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # [CHANGED] Variable 제거, 최신 스타일로
            # ### OLD:
            # data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            # label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            data = data.float().cuda(self.output_device) if isinstance(self.output_device, int) else data.float().cpu()
            label = label.long().cuda(self.output_device) if isinstance(self.output_device, int) else label.long().cpu()
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data)
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0.0
            loss = self.loss(output, label) + l1

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # [CHANGED] 0-dim 텐서 접근 → .item()
            # ### OLD: loss_value.append(loss.data[0])
            loss_value.append(loss.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            # [CHANGED] .item()
            # ### OLD: self.train_writer.add_scalar('loss', loss.data[0], self.global_step)
            self.train_writer.add_scalar('loss', loss.item(), self.global_step)
            self.train_writer.add_scalar('loss_l1', float(l1), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # [CHANGED] 출력 디렉토리 자동 생성
        if wrong_file is not None:
            os.makedirs(os.path.dirname(wrong_file) or ".", exist_ok=True)
            f_w = open(wrong_file, 'w')
        else:
            f_w = None
        if result_file is not None:
            os.makedirs(os.path.dirname(result_file) or ".", exist_ok=True)
            f_r = open(result_file, 'w')
        else:
            f_r = None

        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            process = tqdm(self.data_loader[ln])

            # [CHANGED] volatile 제거 → no_grad()
            with torch.no_grad():
                for batch_idx, (data, label, index) in enumerate(process):
                    # [CHANGED] 최신 스타일로 디바이스 이동
                    if isinstance(self.output_device, int) or (isinstance(self.output_device, str) and str(self.output_device).startswith('cuda')):
                        data  = data.float().cuda(self.output_device, non_blocking=True)
                        label = label.long().cuda(self.output_device, non_blocking=True)
                    else:
                        data  = data.float().cpu()
                        label = label.long().cpu()

                    output = self.model(data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0.0

                    loss = self.loss(output, label)
                    score_frag.append(output.detach().cpu().numpy())
                    # [CHANGED] .item()
                    # ### OLD: loss_value.append(loss.data[0])
                    loss_value.append(loss.item())

                    # optional wrong/right 기록
                    if f_w is not None or f_r is not None:
                        _, predict_label = torch.max(output.data, 1)
                        predict = predict_label.detach().cpu().numpy().tolist()
                        true    = label.detach().cpu().numpy().tolist()
                        for i, x in enumerate(predict):
                            if f_r is not None:
                                f_r.write(f"{x},{true[i]}\n")
                            if f_w is not None and x != true[i]:
                                f_w.write(f"{index[i]},{x},{true[i]}\n")

            score = np.concatenate(score_frag) if len(score_frag) else np.zeros((0, ))
            loss_mean = float(np.mean(loss_value)) if loss_value else 0.0
            accuracy = self.data_loader[ln].dataset.top_k(score, 1) if len(score_frag) else 0.0
            if accuracy > self.best_acc:
                self.best_acc = accuracy
            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss_mean, self.global_step)
                self.val_writer.add_scalar('loss_l1', float(l1), self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score)) if len(score_frag) else {}
            self.print_log('\tMean {} loss of {} batches: {}.'.format(ln, len(self.data_loader[ln]), loss_mean))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(k, 100 * self.data_loader[ln].dataset.top_k(score, k) if len(score_frag) else 0.0))

            if save_score and len(score_frag):
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

        if f_w is not None:
            f_w.close()
        if f_r is not None:
            f_r.close()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-3:
                    break
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            # [CHANGED] wrong/right 파일 상위 폴더 자동 생성
            out_dir = os.path.dirname(self.arg.model_saved_name)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            if not self.arg.test_feeder_args.get('debug', False):
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

def str2bool(v):
    if isinstance(v, bool):  # [CHANGED] 방어코드 추가
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
