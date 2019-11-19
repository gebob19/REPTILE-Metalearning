import torch 
import copy 
import pickle

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt 

from os import path
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
    parser.add_argument('--name', default='default', type=str)
    
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--path', default='default', type=str)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--dsave', action='store_true', default=False)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--n_iterations', default=5000, type=int)

    return parser.parse_args()

def take_n_steps(loss_fcn, optim, model, x, y, n_steps):
    losses = []
    for _ in range(n_steps):
        optim.zero_grad()
        loss = loss_fcn(model(x), y)
        loss.backward()
        optim.step()
        
        losses.append(loss)
    return losses

def get_optimizer(model, lr, optim_state=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0, 0.999))
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)
    return optimizer

class Iter: 
    def __init__(self, path, suffix, train):
        self.path = path 
        self.suffix = suffix
        self.train = train
        self.max = len(list(Path(path).iterdir()))
        self.count = 1
        
    def __iter__(self):
        return self 

    def __next__(self):
        if self.count - 1 == self.max:
            raise StopIteration

        with open(self.path + self.suffix + str(self.count), "rb") as f:   
            self.count += 1
            data = pickle.load(f)
            if self.train: 
                return data[0], data[1]
            else:
                return data[0], data[1], data[2], data[3]


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

    if args.load:
        try:
            model.load_state_dict(torch.load(args.path, map_location=device))
            optim_state = torch.load(args.path+'_optim', map_location=device)
        except Exception:
            print('could not load full model...')

    if args.test: 
        params['outer_iterations'] = 0

    # dont overwrite 
    if path.exists('model_saves/'+model_name):
        model_name = model_name + "_" +str(np.random.randint(100000))

    for outer_i, (train_inputs, train_labels) in tqdm(enumerate(Iter('./train/', 'training', train=True))):

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
        for meta_task_x, meta_task_y in zip(train_inputs, train_labels):
            # new model 
            new_model = model.clone()
            inner_loop_optim = get_optimizer(new_model, params['inner_lr'], optim_state)

            # train on batches of the minidataset
            new_model.train()
            for x, y in zip(meta_task_x, meta_task_y):
                xb = torch.tensor(np.array(x)).unsqueeze(1).to(device)
                yb = torch.tensor(np.array(list(y))).to(device)
                
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

        # update model with avg over mini batches 
        for w in model.parameters():
            w.grad.data.div_(params['meta_batchsize'])
        outter_loop_optim.step()

        # evaluation
        if outer_i % params['validation_rate'] == 0:
            for loader, name in zip([Iter('./evaluation/', 'train/', False), Iter('./evaluation/', 'val/', False)], ['train', 'val']):
                
                for i, (meta_task_x, meta_task_y, test_x, test_y) in enumerate(loader):
                    new_model = model.clone()
                    inner_loop_optim = get_optimizer(new_model, params['inner_lr'], optim_state)

                    for x, y in zip(meta_task_x, meta_task_y):
                        xb = torch.tensor(np.array(x)).unsqueeze(1).to(device)
                        yb = torch.tensor(np.array(list(y))).to(device)
                        
                        take_n_steps(loss_fcn, 
                                    inner_loop_optim,
                                    new_model,
                                    xb, yb, 1)
                    
                    # record optimizer state 
                    optim_state = inner_loop_optim.state_dict()

                    # validation metrics
                    test_x = torch.tensor(np.array(test_x)).unsqueeze(1).to(device)
                    test_y = torch.tensor(np.array(list(test_y))).to(device)

                    new_model.eval()
                    y_preds = new_model(test_x)
                    loss = loss_fcn(y_preds, test_y)
                    accuracy = (y_preds.argmax(-1) == test_y).float().mean()
                                        
                    writer.add_scalar('{}_loss'.format(name), loss, outer_i)
                    writer.add_scalar('{}_acc'.format(name), accuracy, outer_i)

    torch.save(inner_loop_optim.state_dict(), 'model_saves/'+model_name+'_pretest_optim')

    print('testing...')
    n_correct = 0
    n_examples = 0
    for i, (meta_task_x, meta_task_y, test_x, test_y) in enumerate(Iter('./evaluation/', 'test', False)):
        new_model = model.clone()
        inner_loop_optim = get_optimizer(new_model, params['inner_lr'], optim_state)

        for x, y in zip(meta_task_x, meta_task_y):
            xb = torch.tensor(np.array(x)).unsqueeze(1).to(device)
            yb = torch.tensor(np.array(list(y))).to(device)
            
            take_n_steps(loss_fcn, 
                        inner_loop_optim,
                        new_model,
                        xb, yb, 1)

        # record optimizer state 
        optim_state = inner_loop_optim.state_dict()

        # validation metrics
        new_model.eval()
        y_preds = new_model(test_x)
        loss = loss_fcn(y_preds, test_y)
        n_correct += (y_preds.argmax(-1) == test_y).sum().float()
        n_examples += test_x.size(0)

    accuracy = n_correct / n_examples
    print("Accuracy: {}".format(accuracy))

    writer.add_scalar('test_acc', accuracy, 0)
    writer.close()
    print('Summary writer closed...')
 
    print('saving model to {} ...'.format('model_saves/'+model_name))
    torch.save(model.state_dict(), 'model_saves/'+model_name)
    torch.save(inner_loop_optim.state_dict(), 'model_saves/'+model_name+'_optim')

if __name__ == '__main__':
    main()