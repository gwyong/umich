import torch.nn as nn
import torch.nn.init as init
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.eye_(m.weight)
    elif classname.find('Emb') != -1:
        init.normal(m.weight, mean=0, std=0.01)
    
class FCNET(nn.Module):
    def __init__(self, d, ndf, nc):
        super(FCNET, self).__init__()
        
        self.nets = nn.Sequential(
        nn.Conv1d(d, ndf, kernel_size=1, bias=False),
        )

        self.head = nn.Sequential(
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(ndf, nc, kernel_size=1, bias=True),
        )

    def forward(self, input):
        tc = self.nets(input)
        ce = self.head(tc)
        return tc, ce