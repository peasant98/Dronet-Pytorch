import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torchvision.transforms.functional as TF

class DronetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_group, augmentation=False, grayscale=False):
        super(DronetDataset, self).__init__()
        self.root_dir = root_dir
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
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
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

            # labels
        # the resulting list should have all of the paths to the filenames from 
        # self.main_dir
        # go through each folder and get the length
        self.all_img_tuples = img_path_list
    
    def __len__(self):
        '''
        gets the length of the dataset.

        ## parameters:
        
        None.
        '''
        return len(self.all_img_tuples)

    def __getitem__(self, idx):
        '''
        gets an item from the dataset, and returns the whole rgb image and the target

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
        print('Image path: {}'.format(item[0]))
        image = Image.open(item[0]) 
        # data augmentation, can return multiple copies too
        if self.transform != []:
            image = self.transform(image)
        plt.imshow(image)
        plt.title('Steering: {} Collision: {}'.format(target_steer, target_coll))
        plt.show()
        image_tensor = self.to_tensor(image)
        return image_tensor.shape, target_steer, target_coll


dataset = DronetDataset('all-data', 'training', True)
print(len(dataset))
for i in range(100):
    rand_value = np.random.randint(0, len(dataset))
    print(dataset[rand_value])