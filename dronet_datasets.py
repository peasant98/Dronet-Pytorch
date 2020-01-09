import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torchvision.transforms.functional as TF

class DronetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_group, augmentation=False, grayscale=False, verbose=False): 
        '''
        DronetDataset class. Loads data from both Udacity and ETH Zurich bicycle datasets.
        While iterating through this dataset, dataset[i] returns 3 tensors: the normalized image
        tensor, the steering angle (in radians), and the probability of collision.

        ## parameters

        `root_dir`: `str`: path to the root directory.

        `data_group`: `str`: one of either `'training'`, `'testing'`, or `'validation'`.

        `augmentation`: `bool`: whether augmentation is used on images in the dataset. 
        An augmentation consists of a randomly resized crop to `224` by `224` pixels,
        a color jitter with brightness, contrast, and saturation changes, and a random
        (with probability=`0.1`) grayscale selection.

        `grayscale`: `bool`: whether the image will be grayscaled at the end or not

        `verbose`: `bool`: whether to display the image when __iter__() is called in matplotlib,
        along with the steering angle and collision probability being displayed.
        
        '''
        super(DronetDataset, self).__init__()
        self.root_dir = root_dir
        self.verbose = verbose
        self.transforms_list = []
        if augmentation:
            self.transforms_list = [
                transforms.RandomResizedCrop(224),
                # this is a good one, changes
                transforms.ColorJitter(brightness=(0.5,1), contrast=(0.5,1), saturation=(0.7,1)),
                transforms.RandomGrayscale(),
            ]
        if grayscale:
            self.transforms_list.append(transforms.Grayscale())
        # create list of transforms to apply to image
        self.transform = transforms.Compose(self.transforms_list)
        self.to_tensor = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])
        if data_group not in ['training', 'testing', 'validation']:
            raise ValueError('Invalid data group selection. Choose from training, testing, validation')
        self.data_group = data_group
        # get the list of all of the files
        self.main_dir = os.path.join(self.root_dir, self.data_group)
        # get list of folders

        self.folders_list = sorted(os.listdir(self.main_dir))
        # list of sorted folders
        img_path_list = []
        for folder in self.folders_list:
            # print(os.listdir(os.path.join(self.main_dir, folder, 'images')))
            images = sorted(os.listdir(os.path.join(self.main_dir, folder, 'images')))
            exp_type = 'steer'
            if 'labels.txt' in os.listdir(os.path.join(self.main_dir, folder)):
                exp_type = 'coll'
            # list of sorted names
            correct_images = images
            # get np array of labels from steering or collision
            if exp_type == 'coll':
                labels_path = os.path.join(self.main_dir, folder, 'labels.txt')
            else:
                labels_path = os.path.join(self.main_dir, folder, 'steering_{}.txt'.format(data_group))
            # get experiments array from folder
            exp_arr = np.loadtxt(labels_path)
            if images[0][0] >= '0' and images[0][0] <= '9':
                # the case of the udacity data
                correct_images = sorted(images, key=lambda x: int(x[:-4]))
            for ind, img_path in enumerate(correct_images):
                img_path_list.append((os.path.join(self.main_dir, folder, 'images', img_path), exp_type, exp_arr[ind]))
            '''
            each entry in img_path_list contains:

            path from here to the folder

            experiment type

            value of that experiment, either steering angle or collision probability

            '''
        self.all_img_tuples = img_path_list
    
    def __len__(self):
        '''
        gets the length of the dataset. Based on the amount of images.

        ## parameters:
        
        None.
        '''
        return len(self.all_img_tuples)

    def __getitem__(self, idx):
        '''
        gets an item from the dataset, and returns the tensor input and the target

        ## parameters:

        `idx`: the index from which to get the image and its corresponding labels.
        '''
        item = self.all_img_tuples[idx]
        # tuple of steer coll
        target_steer = 0.
        target_coll = 0.
        if item[1] == 'coll':
            target_coll = item[2]
        else:
            target_steer = item[2]
        # get the image
        # print('Image path: {}'.format(item[0]))
        image = Image.open(item[0]) 
        # data augmentation, can return multiple copies too
        if self.transform != []:
            image = self.transform(image)
        if self.verbose:
            plt.imshow(image)
            plt.title('Steering: {} Collision: {}'.format(target_steer, target_coll))
            plt.show()
        image_tensor = self.to_tensor(image)
        target_steer = torch.Tensor([target_steer]).float()
        target_coll = torch.Tensor([target_coll]).float()
        return image_tensor, target_steer, target_coll


# print(len(dataset))
# rand_idx = np.random.randint(0, len(dataset))
# print(dataset[rand_idx])


# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
from dronet_torch import DronetTorch
dronet = DronetTorch(img_dims=(224,224), img_channels=3, output_dim=1)
dronet.to(dronet.device)
dronet.train()
training_dataset = DronetDataset('data/collision/collision_dataset', 'training', True)
validation_dataset = DronetDataset('data/collision/collision_dataset', 'testing', True)

training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=8, 
                                        shuffle=True, num_workers=4)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=2, 
                                        shuffle=True, num_workers=1)


optimizer = torch.optim.Adam(dronet.parameters(), lr=1e-4, weight_decay=1e-5)
idx=0
losses = []
for batch_idx, (img, steer_true, coll_true) in enumerate(training_dataloader):
    epoch = 0
    img_cuda = img.float().to(dronet.device)
    steer_pred, coll_pred = dronet(img_cuda)
    steer_true = steer_true.to(dronet.device)
    coll_true  = coll_true.to(dronet.device)
    loss = dronet.loss(8, steer_true, steer_pred, coll_true, coll_pred)
    loss.backward()
    optimizer.step()
    # zero gradients to prevent accumulation, for now
    optimizer.zero_grad()
    idx+=8
    print(f'Loss: {loss.item()}')
    losses.append(loss.item())
    print(f'Image Count: {idx}')

weights_path = os.path.join('models', 'dronet_trained.pth')
torch.save(dronet.state_dict(), weights_path)
plt.plot(losses)
plt.show()
# need to do:
# transfer learning
# way to get the data
# saving the datas