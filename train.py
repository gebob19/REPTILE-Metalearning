import torch 
import copy 

import numpy as np 
import tensorflow as tf
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


# def take_n_steps(loss_fcn, optim, model, x, y, n_steps):
#     losses = []
#     for _ in range(n_steps):
#         optim.zero_grad()
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

def shuffle_set(x, y):
    # shuffle_idxs = np.random.permutation(x.size(0))
    shuffle_idxs = np.random.permutation(x.shape[0])
    x = x[shuffle_idxs]
    y = y[shuffle_idxs]
    return x, y 

# shuffle 
# src: https://github.com/openai/supervised-reptile/blob/22bda434f0c8c27f1323d9d5c84014e45922ef13/supervised_reptile/reptile.py#L223
def _mini_batches(x, y, batch_size, num_batches):
    current_batch = []
    batch_count = 0
    while True:
        x, y = shuffle_set(x, y)
        for sample in zip(x, y):
            current_batch.append(sample)
            if len(current_batch) < batch_size:
                continue
            xbatch, ybatch = zip(*current_batch)
            # yield torch.stack(xbatch).to(device), torch.stack(ybatch).to(device)
            yield np.stack(xbatch).squeeze(), np.stack(ybatch)
            current_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return


def main():
    debug = True

    args = arg_parser()
    params = way20_params
    param_name = args.name

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(True)

    # init model + optimzers + loss 
    model = TFOmniglotModel(params['n_way'], params['inner_lr'])
    session = tf.Session()
    # only model weights 
    model_state = VariableState(session, tf.trainable_variables())
    # model + optimize params 
    full_state = VariableState(session, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    session.run(tf.global_variables_initializer())

    # setup
    model_name = 'mname={}-nway={}-kshot={}-ntest={}'.format(param_name, params['n_way'], params['k_shots'], params['n_test'])
    writer = SummaryWriter(comment=model_name)

    if args.debug: 
        params['outer_iterations'] = args.n_iterations
    if args.test: 
        params['outer_iterations'] = 0
    # dont overwrite 
    if path.exists('model_saves/'+model_name):
        model_name = model_name + "_" + str(np.random.randint(100000))

    # if args.load:
    #     try:
    #         model.load_state_dict(torch.load(args.path, map_location=device))
    #         optim_state = torch.load(args.path+'_optim', map_location=device)
    #     except Exception:
    #         print('could not load full model...')


    train_loader = get_dataloader('train', params['train_shots'], params['n_way'])

    train_eval_loader = get_dataloader('train', params['k_shots'], params['n_way'])
    val_loader = get_dataloader('val', params['k_shots'], params['n_way'])
    test_loader = get_dataloader('test', params['k_shots'], params['n_way'])

    for outer_i in tqdm(range(params['outer_iterations'])):

        if args.debug and outer_i == args.n_iterations - 1:
            break 

        # lr annealing 
        frac_done = outer_i / params['outer_iterations']
        cur_meta_step_size = frac_done * params['metastep_final'] + (1 - frac_done) * params['outer_lr']

        # # update optimizer step size 
        # for param_group in outter_loop_optim.param_groups:
        #     param_group['lr'] = cur_meta_step_size

        # sample minidataset 
        n_correct, n_examples, loss = 0, 0, 0
        base_model = model_state.export_variables()
        meta_models = []

        for task_i, ((x, y), _) in enumerate(train_loader):
            # new model 
            # new_model = model.clone()
            # inner_loop_optim = get_optimizer(new_model, params['inner_lr'], optim_state)

            # train on batches of the minidataset
            # new_model.train()
            for xb, yb in _mini_batches(x, y, params['inner_batchsize'], params['inner_iterations']):
                session.run(model.minimize_op, feed_dict={model.input_ph: xb,
                                                          model.label_ph: yb})

            meta_models.append(model_state.export_variables())
            model_state.import_variables(base_model)

            # # record optimizer state 
            # optim_state = inner_loop_optim.state_dict()

            # # record weights
            # for w, w_t in zip(model.parameters(), new_model.parameters()):
            #     if w.grad is None:
            #         w.grad = Variable(torch.zeros_like(w)).to(device)
            #     # invert loss eqn. to use descent optimization
            #     w.grad.data.add_(w.data - w_t.data)

            if task_i == params['meta_batchsize'] - 1:
                break

        # update model with avg over mini batches 
        updated_model = interpolate_vars(base_model, average_vars(meta_models), cur_meta_step_size)
        model_state.import_variables(updated_model)

        # evaluation
        if outer_i % params['validation_rate'] == 0:
            for loader, name in zip([train_eval_loader, val_loader], ['train', 'val']):
                # save base variables
                base_variables = full_state.export_variables()
                for task_i, ((x, y), (x_test, y_test)) in enumerate(loader):
                    for xb, yb in _mini_batches(x, y, params['eval_inner_batch'], params['eval_inner_iterations']):
                        session.run(model.minimize_op, feed_dict={model.input_ph: xb,
                                                                 model.label_ph: yb})
                    
                    # record optimizer state 
                    # optim_state = inner_loop_optim.state_dict()

                    # validation metrics
                    ypred, loss = session.run([model.predictions, model.loss], 
                                            feed_dict={model.input_ph: x_test, 
                                                        model.label_ph: y_test})
                    accuracy = (ypred == y_test).astype(np.float32).mean()
                                        
                    writer.add_scalar('{}_loss'.format(name), loss.mean(), outer_i)
                    writer.add_scalar('{}_acc'.format(name), accuracy, outer_i)
                    break
                
                # import base model + optim weights - dont update during val 
                full_state.import_variables(base_variables)

    # torch.save(inner_loop_optim.state_dict(), 'model_saves/'+model_name+'_pretest_optim')

    print('testing...')
    n_correct = 0
    n_examples = 0
    base_model = full_state.export_variables()

    for task_i, ((x, y), (x_test, y_test)) in tqdm(enumerate(test_loader)):
        full_state.import_variables(base_model)

        for xb, yb in _mini_batches(x, y, params['eval_inner_batch'], params['eval_inner_iterations']):
            session.run(model.minimize_op, feed_dict={model.input_ph: xb,
                                                    model.label_ph: yb})

        # record optimizer state 
        # optim_state = inner_loop_optim.state_dict()

        ypred, loss = session.run([model.predictions, model.loss], 
                                    feed_dict={model.input_ph: x_test, 
                                                model.label_ph: y_test})

        bn_correct = (ypred == y_test).astype(np.float32).sum()
        bn_examples = len(y_test)
        
        n_correct += bn_correct
        n_examples += bn_examples

        if task_i == 10000 - 1: 
            break

        # # validation metrics
        # new_model.eval()
        # y_preds = new_model(x_test)
        # loss = loss_fcn(y_preds, y_test)
        # n_correct += (y_preds.argmax(-1) == y_test).sum().float()
        # n_examples += x_test.size(0)

    accuracy = n_correct / n_examples
    print("Accuracy: {}".format(accuracy))

    writer.add_scalar('test_acc', accuracy, 0)
    writer.close()
    print('Summary writer closed...')
 
    # print('saving model to {} ...'.format('model_saves/' + model_name))
    # torch.save(model.state_dict(), 'model_saves/' + model_name)
    # torch.save(inner_loop_optim.state_dict(), 'model_saves/' + model_name + '_optim')

if __name__ == '__main__':
    main()