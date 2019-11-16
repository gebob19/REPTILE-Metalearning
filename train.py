import torch 
import copy 

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt 

from torch.autograd import Variable 
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataloader import *
from params import way5_params, way20_params
from model import OmniglotModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base = Path('data/omniglot/')
data_dir = base/'data'
split_dir = base/'splits'/'vinyals'

# from https://github.com/openai/supervised-reptile/blob/master/supervised_reptile/args.py
import argparse
def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--n_iterations', default=5000, type=int)
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()

# https://github.com/gabrielhuang/reptile-pytorch/blob/master/train_omniglot.py
def make_inf(D):
    while True:
        for x in D:
            yield x

def get_dataloader(split, k_shot, n_way, n_test=1, shuffle=True, inf=True):
    dataset = OmniClassDataset(split=split,
                           data_dir=data_dir, 
                           splits_dir=split_dir,
                           shuffle=shuffle,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               lambda x: 1 - x
                           ]))
    
    dataloader = OmniLoader(k_shot=k_shot, 
                            n_way=n_way,
                            n_test=n_test,
                            dataset=dataset,
                            shuffle=shuffle,
                            pin_memory=True,
                            drop_last=True, 
                            num_workers=8)

    dataloader = make_inf(dataloader) if inf else dataloader
  
    return dataloader

def take_n_steps(loss_fcn, optim, model, x, y, n_steps):
    losses = []
    for _ in range(n_steps):
        optim.zero_grad()
        loss = loss_fcn(model(x), y)
        loss.backward()
        optim.step()
        
        losses.append(loss)
    return losses


def shuffle_set(x, y):
    shuffle_idxs = np.random.permutation(x.size(0))
    x = x[shuffle_idxs]
    y = y[shuffle_idxs]
    return x, y 

def get_optimizer(model, lr, optim_state=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0, 0.999))
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)
    return optimizer

# shuffle 
# src: https://github.com/openai/supervised-reptile/blob/22bda434f0c8c27f1323d9d5c84014e45922ef13/supervised_reptile/reptile.py#L223
def _mini_batches(x, y, batch_size, num_batches):
    cur_batch = []
    batch_count = 0
    while True:
        x, y = shuffle_set(x, y)
        curr_batch = []
        for sample in zip(x, y):
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            xbatch, ybatch = zip(*cur_batch)
            yield torch.stack(xbatch).to(device), torch.stack(ybatch).to(device)
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return


def main():
    debug = True

    args = arg_parser()
    params = way5_params
    param_name = args.name

    # init model + optimzers + loss 
    model = OmniglotModel(n_classes=params['n_way']).to(device)
    outter_loop_optim = torch.optim.SGD(model.parameters(), lr=params['outer_lr'])
    loss_fcn = nn.CrossEntropyLoss()
    optim_state = None 

    # setup
    model_name = 'mname={}-nway={}-kshot={}-ntest={}'.format(param_name, params['n_way'], params['k_shots'], params['n_test'])
    writer = SummaryWriter(comment=model_name)

    # debugging parameters
    if args.debug: 
        params['outer_iterations'] = args.n_iterations

    train_loader = get_dataloader('train', params['train_shots'], params['n_way'])

    train_eval_loader = get_dataloader('train', params['k_shots'], params['n_way'])
    val_loader = get_dataloader('val', params['k_shots'], params['n_way'])
    test_loader = get_dataloader('test', params['k_shots'], params['n_way'], inf=False)

    for outer_i in tqdm(range(params['outer_iterations'])):
        outter_loop_optim.zero_grad()

        # lr annealing 
        frac_done = outer_i / params['outer_iterations']
        cur_meta_step_size = frac_done * params['metastep_final'] + (1 - frac_done) * params['outer_lr']

        # update optimizer step size 
        for param_group in outter_loop_optim.param_groups:
            param_group['lr'] = cur_meta_step_size

        # sample minidataset 
        n_correct, n_examples, loss = 0, 0, 0
        new_vars = []
        for task_i, ((x, y), _) in enumerate(train_loader):
            # new model 
            new_model = model.clone()
            inner_loop_optim = get_optimizer(new_model, params['inner_lr'], optim_state)

            # train on batches of the minidataset
            new_model.train()
            for xb, yb in _mini_batches(x, y, params['inner_batchsize'], params['inner_iterations']):
                take_n_steps(loss_fcn, 
                                inner_loop_optim,
                                new_model,
                                xb, yb, 1)

            # record optimizer state 
            optim_state = inner_loop_optim.state_dict()

            # record weights
            for w, w_t in zip(model.parameters(), new_model.parameters()):
                if w.grad is None:
                    w.grad = Variable(torch.zeros_like(w)).to(device)
                # invert loss eqn. to use descent optimization
                w.grad.data.add_(w.data - w_t.data)

            if task_i == params['meta_batchsize'] - 1:
                break

        # update model with avg over mini batches 
        for w in model.parameters():
            w.grad.data.div_(params['meta_batchsize'])
        outter_loop_optim.step()

        # evaluation
        if outer_i % params['validation_rate'] == 0:
            for loader, name in zip([train_eval_loader, val_loader], ['train', 'val']):
                for task_i, ((x, y), (x_test, y_test)) in enumerate(loader):
                    new_model = model.clone()
                    # dont restore optim state - info leakage 
                    inner_loop_optim = get_optimizer(new_model, params['inner_lr'], optim_state)

                    new_model.train()
                    for xb, yb in _mini_batches(x, y, params['eval_inner_batch'], params['eval_inner_iterations']):
                        take_n_steps(loss_fcn, 
                                    inner_loop_optim,
                                    new_model,
                                    xb, yb, 1)
                    
                    # validation metrics
                    new_model.eval()
                    y_preds = new_model(x_test)
                    loss = loss_fcn(y_preds, y_test)
                    accuracy = (y_preds.argmax(-1) == y_test).float().mean()
                                        
                    writer.add_scalar('{}_loss'.format(name), loss, outer_i)
                    writer.add_scalar('{}_acc'.format(name), accuracy, outer_i)
                    break

    print('testing...')
    n_correct = 0
    n_examples = 0
    for task_i, ((x, y), (x_test, y_test)) in tqdm(enumerate(test_loader)):
        new_model = model.clone()
        inner_loop_optim = get_optimizer(new_model, params['inner_lr'], optim_state)

        new_model.train()
        for xb, yb in _mini_batches(x, y, params['eval_inner_batch'], params['eval_inner_iterations']):
            take_n_steps(loss_fcn, 
                        inner_loop_optim,
                        new_model,
                        xb, yb, 1)

        # validation metrics
        new_model.eval()
        y_preds = new_model(x_test)
        loss = loss_fcn(y_preds, y_test)
        n_correct += (y_preds.argmax(-1) == y_test).sum().float()
        n_examples += x_test.size(0)

    accuracy = n_correct / n_examples
    print("Accuracy: {}".format(accuracy))

    writer.add_scalar('test_acc', accuracy, 0)
    writer.close()
    print('Summary writer closed...')

    print('saving model to {} ...'.format('model_saves/'+model_name))
    torch.save(model.state_dict(), 'model_saves/'+model_name)

if __name__ == '__main__':
    main()