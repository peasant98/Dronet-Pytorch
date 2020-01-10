import sys
import numpy as np
import os

import dronet_torch
from dronet_datasets import DronetDataset

import torch


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
                epochs, batch_size, steps_save, k):
    '''
    trains the model.

    ## parameters:
    

    '''
    model.to(model.device)

    model.train()
    # create dataloaders for validation and training
    training_dataset = DronetDataset('data/collision/collision_dataset', 'training', True)
    validation_dataset = DronetDataset('data/collision/collision_dataset', 'testing', True)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=4)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=4, 
                                            shuffle=False, num_workers=4)

    # adam optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epoch_loss = np.zeros((epochs, 2))
    for epoch in range(epochs):
        # scale the weights on the loss and the epoch number
        train_losses = []
        validation_losses = []
        # rip through the dataset
        other_val = (1 - torch.exp(torch.Tensor([-1*model.decay * (epoch-10)]))).float().to(model.device)
        model.beta = torch.max(torch.Tensor([0]).float().to(model.device), other_val)
        for batch_idx, (img, steer_true, coll_true) in enumerate(training_dataloader):
            img_cuda = img.float().to(model.device)
            steer_pred, coll_pred = model(img_cuda)

            # get loss, perform hard mining
            steer_true = steer_true.to(model.device)
            coll_true  = coll_true.to(model.device)

            loss = model.loss(k, steer_true, steer_pred, coll_true, coll_pred)
            # backpropagate loss
            loss.backward()
            # optimizer step
            optimizer.step()
            # zero gradients to prevent accumulation, for now
            optimizer.zero_grad()
            train_losses.append(loss.item())
            print(f'Training Images Epoch {epoch}: {batch_idx * batch_size}')
        train_loss = np.array(train_losses).mean()
        if epoch % steps_save == 0:
            print('Saving results...')

            weights_path = os.path.join('models', f'weights_{epoch:03d}.pth')
            torch.save(model.state_dict(), weights_path)
        # evaluate on validation set
        for batch_idx, (img, steer_true, coll_true) in enumerate(validation_dataloader):
            img_cuda = img.float().to(model.device)
            steer_pred, coll_pred = model(img_cuda)
            steer_true = steer_true.to(model.device)
            coll_true = coll_true.to(model.device)
            loss = model.loss(k, steer_true, steer_pred, coll_true, coll_pred)
            validation_losses.append(loss.item())
            print(f'Validation Images: {batch_idx * 4}')

        validation_loss = np.array(validation_losses).mean()
        epoch_loss[epoch, 0] = train_loss 
        epoch_loss[epoch, 1] = validation_loss
        # Save training and validation losses.
    # save final results
    weights_path = os.path.join('models', 'dronet_trained.pth')
    torch.save(model.state_dict(), weights_path)
    np.savetxt(os.path.join('models', 'losses.txt'), epoch_loss)

if __name__ == "__main__":
    dronet = getModel((224,224), 3, 1, None)
    print(dronet)
    trainModel(dronet, 256, 16, 5, 8)
