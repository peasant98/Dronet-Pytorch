import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# dronet implementation in pytorch.
class DronetTorch(nn.Module):
    def __init__(self, img_dims, img_channels, output_dim):
        """
        Define model architecture.
        
        ## Arguments

        `img_dim`: image dimensions.

        `img_channels`: Target image channels.

        `output_dim`: Dimension of model output.

        """
        super(DronetTorch, self).__init__()
        self.img_dims = img_dims
        self.channels = img_channels
        self.output_dim = output_dim
        self.conv_modules = nn.ModuleList()
        self.alpha = torch.Tensor([1]).float()
        self.beta = torch.Tensor([0]).float()

        # Initialize number of samples for hard-mining
        self.k_mse = torch.Tensor([2]).int()
        self.k_entrpoy = torch.Tensor([2]).int()


        self.conv_modules.append(nn.Conv2d(self.channels, 32, (5,5), stride=(2,2), padding=(2,2)))
        filter_amt = np.array([32,64,128])
        for f in filter_amt:
            x1 = int(f/2) if f!=32 else f
            x2 = f
            self.conv_modules.append(nn.Conv2d(x1, x2, (3,3), stride=(2,2), padding=(1,1)))
            self.conv_modules.append(nn.Conv2d(x2, x2, (3,3), padding=(1,1)))
            self.conv_modules.append(nn.Conv2d(x1, x2, (1,1), stride=(2,2)))
        # create convolutional modules
        self.maxpool1 = nn.MaxPool2d((3,3), (2,2))

        bn_amt = np.array([32,32,32,64,64,128])
        self.bn_modules = nn.ModuleList()
        for i in range(6):
            self.bn_modules.append(nn.BatchNorm2d(bn_amt[i]))

        self.relu_modules = nn.ModuleList()
        for i in range(7):
            self.relu_modules.append(nn.ReLU())
        self.dropout1 = nn.Dropout()

        self.linear1 = nn.Linear(6272, output_dim)
        self.linear2 = nn.Linear(6272, output_dim)
        self.sigmoid1 = nn.Sigmoid()
        self.init_weights()
        self.decay = 0.1

        

    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.conv_modules[1].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[2].weight)

        torch.nn.init.kaiming_normal_(self.conv_modules[4].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[5].weight)

        torch.nn.init.kaiming_normal_(self.conv_modules[7].weight)
        torch.nn.init.kaiming_normal_(self.conv_modules[8].weight)

    def forward(self, x, targets=None):
        '''
        forward pass of dronet
        '''
        bn_idx = 0
        conv_idx = 1
        relu_idx = 0

        x = self.conv_modules[0](x)
        x1 = self.maxpool1(x)
        
        for i in range(3):
            x2 = self.bn_modules[bn_idx](x1)
            x2 = self.relu_modules[relu_idx](x2)
            x2 = self.conv_modules[conv_idx](x2)
            x2 = self.bn_modules[bn_idx+1](x2)
            x2 = self.relu_modules[relu_idx+1](x2)
            x2 = self.conv_modules[conv_idx+1](x2)
            x1 = self.conv_modules[conv_idx+2](x1)
            x3 = torch.add(x1,x2)
            x1 = x3
            bn_idx+=2
            relu_idx+=2
            conv_idx+=3

        x4 = torch.flatten(x3).reshape(-1, 6272)
        x4 = self.relu_modules[-1](x4)
        x5 = self.dropout1(x4)

        steer = self.linear1(x5)

        collision = self.linear2(x5)
        collision = self.sigmoid1(collision)

        return steer, collision

    def loss(self, k, steer_true, steer_pred, coll_true, coll_pred):
        # for steering angle
        mse_loss = self.alpha * (self.hard_mining_mse(k, steer_true, steer_pred))
        # for collision probability
        bce_loss = self.beta * (self.hard_mining_entropy(k, coll_true, coll_pred))
        return mse_loss + bce_loss

    def hard_mining_mse(self, k, y_true, y_pred):
        '''
        Compute Mean Square Error for steering 
        evaluation and hard-mining for the current batch.

        ### parameters
        
        `k`: `int`: number of samples for hard-mining
        '''
        # Parameter t indicates the type of experiment
        t = y_true[:,0]
        # no. of steering samples
        samples_steer = torch.eq(t,1).int()
        n_samples_steer = torch.sum(samples_steer)
        if n_samples_steer == 0:
            return 0.
        else:
            # predicted and real steerings
            pred_steer = torch.squeeze(y_pred, -1)
            true_steer = y_true[:,1]
            # steering loss
            loss_steer = torch.mul(t, (pred_steer - true_steer)**2)

            # hard mining
            k_min = torch.min(k, n_samples_steer)
            _, indices = torch.topk(loss_steer, k=k_min)

            max_loss_steer = torch.gather(loss_steer, dim=1, index=indices)

            hard_loss_steer = torch.div(torch.sum(max_loss_steer), k.float())
            return hard_loss_steer

    def hard_mining_entropy(self, k, y_true, y_pred):
        '''
        computes binary cross entropy for probability collisions and hard-mining.

        ## parameters

        `k`: number of samples for hard-mining
        '''
        # parameter indicates the type of experiment
        t = y_true[:,0]

        # number of collision samples
        samples_coll = torch.eq(t,0).int()
        n_samples_coll = torch.sum(samples_coll)

        if n_samples_coll == 0:
            return 0.
        else:
            # predicted and real labels
            pred_coll = torch.squeeze(y_pred, -1)
            true_coll = y_true[:,1]
            # collision loss
            loss_coll = torch.mul((1-t), F.binary_cross_entropy(pred_coll, true_coll))
            # hard mining
            k_min = torch.min(k, n_samples_coll)
            _, indices = torch.topk(loss_coll, k=k_min)
            max_loss_coll = torch.gather(loss_coll, dim=1, index=indices)
            hard_loss_coll = torch.div(torch.sum(max_loss_coll), k.float())
            return hard_loss_coll


# one dim for steering angle, another for prob. of collision
dronet = DronetTorch(img_dims=(224,224), img_channels=3, output_dim=1)
dronet.cuda()
m = torch.ones((1,3, 224, 224)).cuda()
res = dronet(m)
print(res)