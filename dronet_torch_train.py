import sys
import numpy as np
import os

import dronet_torch
from dronet_datasets import DronetDataset

import torch

def train(img_dim, img_channels, output_channels, weights_path=None):
    # create dronet model
    dronet = dronet_torch.DronetTorch(img_dim, img_channels, output_channels)

    if weights_path != None:
        try:
            dronet.load_state_dict(torch.load(weights_path))
        except:
            print('Invalid weights path')
    else:
        print('No weights path found, model is untrained.')
    

def getModel(img_dims, img_channels, output_dim, weights_path):
    '''
      Initialize model.

      ## Arguments

        `img_dims`: Target image dimensions.

        `img_channels`: Target image channels.

        `output_dim`: Dimension of model output.
        
        `weights_path`: Path to pre-trained model.

      ## Returns
        `model`: the pytorch model
    '''
    model = dronet_torch.DronetTorch(img_dims, img_channels, output_dim)
    # if weights path exists...
    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path))
            print("Loaded model from {}".format(weights_path))
        except:
            print("Impossible to find weight path. Returning untrained model")

    return model

def trainModel(model: dronet_torch.DronetTorch, 
                epochs, batch_size, steps_save):
    '''
    trains the model.

    ## parameters:
    

    '''
    model.to(model.device)

    model.train()
    # create dataloaders for validation and training
    training_dataset = DronetDataset('data/collision/collision_dataset', 'training', True)
    validation_dataset = DronetDataset('data/collision/collision_dataset', 'testing', True)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=2, 
                                            shuffle=True, num_workers=1)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=2, 
                                            shuffle=True, num_workers=1)

    # adam optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # value of k for hard mining
    k = 2
    for epoch in range(epochs):
        # Ltot=LMSE+max(0,1−exp−decay(epoch−epoch0))LBCE
        # scale the weights on the loss and the epoch number
        
        # rip through the dataset
        other_val = (1 - torch.exp(torch.Tensor([-1*model.decay * (epoch-10)]))).float().cuda()
        model.beta = torch.max(torch.Tensor([0]).float().cuda(), other_val)
        for batch_idx, (img, steer_true, coll_true) in enumerate(training_dataloader):
            img_cuda = img.float().cuda()
            steer_pred, coll_pred = model(img_cuda)

            # get loss, perform hard mining
            loss = model.loss(k, steer_true, steer_pred, coll_true, coll_pred)
            # backpropagate loss
            loss.backward()
            # optimizer step
            optimizer.step()
            # zero gradients to prevent accumulation, for now
            optimizer.zero_grad()

        if epoch % steps_save == 0:
            # save model and run validation


            # for (img, steer_true, coll_true) in validation_dataloader:
            #     img_cuda = img.float().cuda()
            #     steer_pred, coll_pred = model(img_cuda)
            #     loss = model.loss(k, steer_true, steer_pred, coll_true, coll_pred)


            weights_path = os.path.join('checkpoints', 'weights_{epoch:03d}.pth')
            torch.save(model.state_dict(), weights_path)
        # evaluate on validation set
        

        # Save training and validation losses.
    # save final results
    weights_path = os.path.join('models', 'dronet_trained.pth')
    torch.save(model.state_dict(), weights_path)

if __name__ == "__main__":
    dronet = getModel((224,224), 3, 1, None)
    print(dronet)
    trainModel(dronet, 1, 2, 100)
