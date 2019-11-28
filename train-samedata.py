import torch 
import copy 
import pickle

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt 

import tensorflow as tf
from functools import partial

from os import path
from torch.autograd import Variable 
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataloader import *
from params import way5_params, way20_params
from model import OmniglotModel, TFOmniglotModel

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

# def take_n_steps(loss_fcn, optim, model, x, y, n_steps):
#     losses = []
#     for _ in range(n_steps):
#         optim.zero_grad()
#         print(model(x))
#         loss = loss_fcn(model(x), y)
#         loss.backward()
#         optim.step()
        
#         losses.append(loss)
#     return losses

# def get_optimizer(model, lr, optim_state=None):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0, 0.999))
#     if optim_state is not None:
#         optimizer.load_state_dict(optim_state)
#     return optimizer

# def get_metrics(model, loss_fcn, x, y):
#     model.eval()

#     y_preds = model(x)
#     loss = loss_fcn(y_preds, y)
#     n_correct = (y_preds.argmax(-1) == y).sum().float()
#     n_examples = x.size(0)

#     return loss, n_correct, n_examples

# def meta_train(meta_task_x, meta_task_y, test_x, test_y, model, loss_fcn, optim, train, name):
    
#     model.train()
#     for x, y in zip(meta_task_x, meta_task_y):
#         xb = torch.tensor(np.array(x)).unsqueeze(1).to(device)
#         yb = torch.tensor(np.array(list(y))).to(device)
        
#         take_n_steps(loss_fcn, 
#                         inner_loop_optim,
#                         new_model,
#                         xb, yb, 1)
    
#     # validation metrics
#     if not train: 
#         test_x = torch.tensor(np.array(test_x)).unsqueeze(1).to(device)
#         test_y = torch.tensor(np.array(list(test_y))).to(device)

#         loss, accuracy = get_metrics(model, loss_fcn, test_x, test_y)
                            
#         if name in ['val', 'train']:
#             writer.add_scalar('{}_loss'.format(name), loss, outer_i)
#             writer.add_scalar('{}_acc'.format(name), accuracy, outer_i)
def interpolate_vars(old_vars, new_vars, epsilon):
    """
    Interpolate between two sequences of variables.
    """
    return add_vars(old_vars, scale_vars(subtract_vars(new_vars, old_vars), epsilon))

def average_vars(var_seqs):
    """
    Average a sequence of variable sequences.
    """
    res = []
    for variables in zip(*var_seqs):
        res.append(np.mean(variables, axis=0))
    return res

def subtract_vars(var_seq_1, var_seq_2):
    """
    Subtract one variable sequence from another.
    """
    return [v1 - v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def add_vars(var_seq_1, var_seq_2):
    """
    Add two variable sequences.
    """
    return [v1 + v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def scale_vars(var_seq, scale):
    """
    Scale a variable sequence.
    """
    return [v * scale for v in var_seq]


class VariableState:
    """
    Manage the state of a set of variables.
    """
    def __init__(self, session, variables):
        self._session = session
        self._variables = variables
        self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                              for v in variables]
        assigns = [tf.assign(v, p) for v, p in zip(self._variables, self._placeholders)]
        self._assign_op = tf.group(*assigns)

    def export_variables(self):
        """
        Save the current variables.
        """
        return self._session.run(self._variables)

    def import_variables(self, values):
        """
        Restore the variables.
        """
        self._session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, values)))


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
        if self.count == self.max - 1:
            raise StopIteration

        print('reading... {}'.format(self.path + self.suffix + str(self.count)))
        with open(self.path + self.suffix + str(self.count), "rb") as f:   
            self.count += 1
            data = pickle.load(f)
            if self.train: 
                return data[0], data[1]
            else:
                return data[0], data[1], data[2], data[3]


def main():
    args = arg_parser()
    params = way5_params
    param_name = args.name

    # init model + optimzers + loss 
    model = TFOmniglotModel(params['n_way'], params['inner_lr'])
    session = tf.Session()
    # only model weights 
    model_state = VariableState(session, tf.trainable_variables())
    # model + optimize params 
    full_state = VariableState(session, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    session.run(tf.global_variables_initializer())


    # outter_loop_optim = torch.optim.SGD(model.parameters(), lr=params['outer_lr'])
    # loss_fcn = nn.CrossEntropyLoss()
    # optim_state = None 

    # setup
    model_name = 'mname={}-nway={}-kshot={}-ntest={}'.format(param_name, params['n_way'], params['k_shots'], params['n_test'])
    writer = SummaryWriter(comment=model_name)

    train_eval, val_eval = Iter('./evaluation/train/', '', False), Iter('./evaluation/val/', '', False)

    # debugging parameters
    if args.debug: 
        params['outer_iterations'] = args.n_iterations

    # if args.load:
    #     try:
    #         model.load_state_dict(torch.load(args.path, map_location=device))
    #         optim_state = torch.load(args.path+'_optim', map_location=device)
    #     except Exception:
    #         print('could not load full model...')

    # load model weights 
    # with open("./src_modelweights/training"+str(-1), "rb") as f:            
    #     src_modelweights = pickle.load(f)
    # for i, (sw, p) in enumerate(zip(src_modelweights, model.parameters())):
    #     if i in [0, 4, 8, 12]:
    #         p.data = Variable(torch.tensor(sw.transpose(-1, -2, 0, 1)))
    #     elif i == len(src_modelweights)-2:
    #         p.data = Variable(torch.tensor(sw.transpose(1, 0)))
    #     else:
    #         p.data = Variable(torch.tensor(sw))
    # print('loaded model weights...')

    # mparams = [p for p in model.parameters()]
    # with open("./modelweights/training"+str(-1), "wb") as f:            
    #         pickle.dump(mparams, f)
    
    if args.test: 
        params['outer_iterations'] = 0

    # dont overwrite 
    if path.exists('model_saves/'+model_name):
        model_name = model_name + "_" +str(np.random.randint(100000))

    for outer_i, (train_inputs, train_labels) in tqdm(enumerate(Iter('./train/', 'training', train=True))):
        frac_done = outer_i / params['outer_iterations']
        cur_meta_step_size = frac_done * params['metastep_final'] + (1 - frac_done) * params['outer_lr']
        
        base_model = model_state.export_variables()
        meta_models = []
        for meta_task_x, meta_task_y in zip(train_inputs, train_labels):    
            for x, y in zip(meta_task_x, meta_task_y):
                session.run(model.minimize_op, feed_dict={model.input_ph: x,
                                                        model.label_ph: y})

            meta_models.append(model_state.export_variables())
            model_state.import_variables(base_model)
        
        # take a step 
        updated_model = interpolate_vars(base_model, average_vars(meta_models), cur_meta_step_size)
        model_state.import_variables(updated_model)
        
        if outer_i % params['validation_rate'] == 0:
            for loader, name in zip([train_eval, val_eval], ['train', 'val']):
                # dont update the optimizer 
                base_model = full_state.export_variables()
                for i, (meta_task_x, meta_task_y, test_x, test_y) in enumerate(loader):
                    for x, y in zip(meta_task_x, meta_task_y):
                        session.run(model.minimize_op, feed_dict={model.input_ph: x,
                                                                model.label_ph: y})

                    ypred, loss = session.run([model.predictions, model.loss], 
                                            feed_dict={model.input_ph: test_x, 
                                                        model.label_ph: test_y})

                    accuracy = (ypred == test_y).astype(np.float32).mean()
                    
                    writer.add_scalar('{}_loss'.format(name), loss.mean(), outer_i)
                    writer.add_scalar('{}_acc'.format(name), accuracy, outer_i)

    # evaluate on test 
    n_correct = 0
    n_examples = 0
    for i, (meta_task_x, meta_task_y, test_x, test_y) in tqdm(enumerate(Iter('./evaluation/test/', '', False))):
        base_model = full_state.export_variables()
        for x, y in zip(meta_task_x, meta_task_y):
            session.run(model.minimize_op, feed_dict={model.input_ph: x,
                                                    model.label_ph: y})

        ypred, loss = session.run([model.predictions, model.loss], 
                                feed_dict={model.input_ph: test_x, 
                                            model.label_ph: test_y})

        bn_correct = (ypred == test_y).astype(np.float32).sum()
        bn_examples = len(test_y)
        
        n_correct += bn_correct
        n_examples += bn_examples

    accuracy = n_correct / n_examples

    print("Accuracy: {}".format(accuracy))
    writer.add_scalar('test_acc', accuracy, 0)
    writer.close()
    print('Summary writer closed...')
            

    # for outer_i, (train_inputs, train_labels) in tqdm(enumerate(Iter('./train/', 'training', train=True))):

    #     outter_loop_optim.zero_grad()

    #     # lr annealing 
    #     frac_done = outer_i / params['outer_iterations']
    #     cur_meta_step_size = frac_done * params['metastep_final'] + (1 - frac_done) * params['outer_lr']

    #     # update optimizer step size 
    #     for param_group in outter_loop_optim.param_groups:
    #         param_group['lr'] = cur_meta_step_size

    #     # sample minidataset 
    #     n_correct, n_examples, loss = 0, 0, 0
    #     new_vars = []
    #     for meta_task_x, meta_task_y in zip(train_inputs, train_labels):
    #         # new model 
    #         new_model = model.clone()
    #         inner_loop_optim = get_optimizer(new_model, params['inner_lr'], optim_state)


    #         # train on batches of the minidataset
    #         new_model.train()
    #         for x, y in zip(meta_task_x, meta_task_y):
    #             xb = torch.tensor(np.array(x)).unsqueeze(1).to(device)
    #             yb = torch.tensor(np.array(list(y))).to(device)
                
    #             take_n_steps(loss_fcn, 
    #                          inner_loop_optim,
    #                          new_model,
    #                          xb, yb, 1)

    #         # record optimizer state 
    #         optim_state = inner_loop_optim.state_dict()

    #         # record weights
    #         for w, w_t in zip(model.parameters(), new_model.parameters()):
    #             if w.grad is None:
    #                 w.grad = Variable(torch.zeros_like(w)).to(device)
    #             # invert loss eqn. to use descent optimization
    #             w.grad.data.add_(-w_t.data)


    #     grad = [-p.grad.data / params['meta_batchsize'] for p in model.parameters()]
    #     with open('./metabatch_avg_weights/training'+str(outer_i), "wb") as f:
    #         pickle.dump(grad, f)

    #     # update model with avg over mini batches 
    #     for w in model.parameters():
    #         w.grad.data.div_(params['meta_batchsize'])
    #         w.grad.data.add_(w.data)

    #     outter_loop_optim.step()
        
    #     grad = [p.grad.data for p in model.parameters()]
    #     mparams = [p for p in model.parameters()]
    #     with open('./gradients/training'+str(outer_i), "wb") as f:
    #         pickle.dump(grad, f)
    #     with open("./train_data/training"+str(outer_i), "wb") as f:            
    #         pickle.dump([train_inputs, train_labels], f)
    #     with open("./modelweights/training"+str(outer_i), "wb") as f:            
    #         pickle.dump(mparams, f)


    #     # evaluation
    #     if outer_i % params['validation_rate'] == 0:
    #         for loader, name in zip([train_eval, val_eval], ['train', 'val']):
                
    #             for i, (meta_task_x, meta_task_y, test_x, test_y) in enumerate(loader):
    #                 new_model = model.clone()
    #                 inner_loop_optim = get_optimizer(new_model, params['inner_lr'], optim_state)

    #                 for x, y in zip(meta_task_x, meta_task_y):
    #                     xb = torch.tensor(np.array(x)).unsqueeze(1).to(device)
    #                     yb = torch.tensor(np.array(list(y))).to(device)
                        
    #                     take_n_steps(loss_fcn, 
    #                                 inner_loop_optim,
    #                                 new_model,
    #                                 xb, yb, 1)

    #                 # validation metrics
    #                 test_x = torch.tensor(np.array(test_x)).unsqueeze(1).to(device)
    #                 test_y = torch.tensor(np.array(list(test_y))).to(device)
    #                 loss, n_correct, n_examples = get_metrics(new_model, loss_fcn, test_x, test_y)
    #                 accuracy = n_correct / n_examples

    #                 writer.add_scalar('{}_loss'.format(name), loss, outer_i)
    #                 writer.add_scalar('{}_acc'.format(name), accuracy, outer_i)

    #                 break

    # print('saving model to {} ...'.format('model_saves/'+model_name))
    # torch.save(model.state_dict(), 'model_saves/'+model_name)
    # torch.save(inner_loop_optim.state_dict(), 'model_saves/'+model_name+'_pretest_optim')

    # print('testing...')
    # n_correct = 0
    # n_examples = 0
    # for i, (meta_task_x, meta_task_y, test_x, test_y) in tqdm(enumerate(Iter('./evaluation/test/', '', False))):
    #     new_model = model.clone()
    #     inner_loop_optim = get_optimizer(new_model, params['inner_lr'], optim_state)

    #     for x, y in zip(meta_task_x, meta_task_y):
    #         xb = torch.tensor(np.array(x)).unsqueeze(1).to(device)
    #         yb = torch.tensor(np.array(list(y))).to(device)
            
    #         take_n_steps(loss_fcn, 
    #                     inner_loop_optim,
    #                     new_model,
    #                     xb, yb, 1)

    #     test_x = torch.tensor(np.array(test_x)).unsqueeze(1).to(device)
    #     test_y = torch.tensor(np.array(list(test_y))).to(device)
    #     # validation metrics
    #     loss, bn_correct, bn_examples = get_metrics(new_model, loss_fcn, test_x, test_y)
    #     n_correct += bn_correct
    #     n_examples += bn_examples

    # accuracy = n_correct / n_examples
    # print("Accuracy: {}".format(accuracy))

    # writer.add_scalar('test_acc', accuracy, 0)
    # writer.close()
    # print('Summary writer closed...')
 
    print('saving model to {} ...'.format('model_saves/'+model_name))
    torch.save(model.state_dict(), 'model_saves/'+model_name)
    torch.save(inner_loop_optim.state_dict(), 'model_saves/'+model_name+'_optim')

if __name__ == '__main__':
    main()