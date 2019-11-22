import torch.nn as nn 
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OmniglotModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self. n_classes = n_classes

        conv_block = lambda in_dim:(nn.Conv2d(in_dim, 64, 3, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        
        self.cnn = nn.Sequential(
            *conv_block(1),
            *conv_block(64),
            *conv_block(64), 
            *conv_block(64)
        )
        self.linear = nn.Sequential(
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

    def clone(self):
        clone = OmniglotModel(self.n_classes)
        clone.load_state_dict(self.state_dict())
        return clone.to(device)