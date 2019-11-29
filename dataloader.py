import torch 
import random

import numpy as np 
import pandas as pd 
import torch.utils.data as data

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OmniClassDataset(data.Dataset):
    """
    Pytorch wrapper for Dataset for Omniglot.
    
    ----------------------
    
    split: must be one in ['train', 'val', 'test']
    splits_dir: directory which contains datasplit .txt files - data/omniglot/splits/vinyals/
    data_dir: directory which contains the omniglot dataset - data/omniglot/data
    transform: extra Pytorch transformations to apply - transforms.Compose() instance
    shuffle: shuffle data before returning
    
    ----------------------
    
    See OmniLoader for full example...
    
    """
    def __init__(self, split, splits_dir, data_dir, transform=None, shuffle=True):
        assert split in ['train', 'val', 'test'], "Invalid Split: {}".format(split)
        self.dataset = pd.read_csv(splits_dir/(split+'.txt'), sep='\n', header=None)
        self.dataset.columns = ['dataset']
        
        self.transform = transform
        self.data_dir = data_dir
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns entire dataset for a meta-task at idx of split.txt
        """
        dataset = self.dataset['dataset'].values[idx]
        
        # dset = path/rot{000, 090, ...}
        dset = dataset.split('/')
        path, rot = '/'.join(dset[:-1]), float(dset[-1][3:])
        
        # read, resize, rotate dataset
        D = list((self.data_dir/path).iterdir())
        D = [Image.open(x).resize((28,28), resample=Image.LANCZOS).rotate(rot) for x in D]
        
        if self.transform is not None:
            D = [self.transform(x) for x in D]
        
        if self.shuffle:
            random.shuffle(D)
            
        D = torch.stack(D)
        return D
    
class OmniLoader(data.DataLoader):
    """
    Pytorch Dataloader for Omniglot - use with Omniglot dataset object.
    
    ----------------------
    
    k_shot: model will be meta trained on k examples of each class 
    n_way: model will classify with n possible classes 
    n_test: number of examples per class to be tested on 
    dataset: OmniClassDataset instance
    kwargs: Pytorch dataloader specific parameters
    
    ----------------------
    
    base = Path('data/omniglot/')
    data_dir = base/'data'
    split_dir = base/'splits'/'vinyals'
    
    dataset = OmniClassDataset(split=split,
                       data_dir=data_dir, 
                       splits_dir=split_dir,
                       shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           lambda x: 1 - x
                       ]))

    dataloader = OmniLoader(k_shot=k_shot, 
                            n_way=n_way,
                            n_test=n_test,
                            dataset=dataset,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=8)
                        
    for i, ((train_x, train_y), (test_x, test_y)) in enumerate(dataloader):    
        break
    
    """
    def __init__(self, k_shot, n_way, n_test, dataset, shuffle, **kwargs):
        self.n_way = n_way
        self.k_shot = k_shot
        self.shuffle = shuffle
        self.n_test = n_test
        self.base_dl = data.DataLoader(dataset, batch_size=n_way, shuffle=shuffle, **kwargs)
    
    def shuffle_set(self, x, y):
        shuffle_idxs = np.random.permutation(x.size(0))
        x = x[shuffle_idxs]
        y = y[shuffle_idxs]
        return x, y 

    def D_to_xy(self, D):
        """
        Converts dataset object D to (*, 1, 28, 28) with corresponding labels.
        """
        # Train: n * k_shot 
        # Test:  n * (n_examples - k_shot)
        x = D.contiguous().view((D.size(0) * D.size(1), 1, 28, 28))#.to(device)
        y = torch.from_numpy(np.array([[i] * D.size(1) for i in range(self.n_way)]).flatten())#.to(device)
        
        assert x.size(0) == y.size(0)

        # shuffle 
        if self.shuffle: 
            x, y = self.shuffle_set(x, y)
        
        return x, y
        
    def __iter__(self):
        # (n_way, n_examples, 1, 28, 28)
        for task_data in self.base_dl:
            x_train, y_train = self.D_to_xy(task_data[:, :self.k_shot])

            k_test_data = task_data[:, self.k_shot:]
            # shuffle class examples & take n_test examples for each 
            k_test_data = k_test_data[:, torch.randperm(k_test_data.size(1))][:, :self.n_test]

            k_test_labels = torch.tensor(np.array([[i] * self.n_test for i in range(k_test_data.size(0))]))
            
            idx = torch.randperm(self.n_way * self.n_test)
            x_test, y_test = k_test_data.reshape(-1, 1, 28, 28)[idx], k_test_labels.reshape(-1)[idx]

            train = (v.numpy().squeeze() for v in [x_train, y_train])
            test = (v.numpy().squeeze() for v in [x_test, y_test])
            
            yield train, test
            # yield (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))

