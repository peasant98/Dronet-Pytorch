import numpy as np
import torch

import os
import pandas as pd
from enum import Enum

# data module for dronet.

def rand_split(data_path_folder):
    # need to move the data into correct spot first
    csv_path = os.path.join(data_path_folder, 'interpolated.csv')
    # results = os.listdir(data_path_folder)
    interpolated_df = pd.read_csv(csv_path, usecols=[1,2,3,4,5,6,7,8])
    # get the frames that are center camera
    correct_interpolated_df = interpolated_df[interpolated_df['frame_id']=='center_camera']
    correct_interpolated_df['filename'] = correct_interpolated_df['filename'].apply(lambda x: 'images'+x[6:])

    img_list = os.listdir(os.path.join(data_path_folder, 'images'))
    real_indices = ((correct_interpolated_df.index.values - 2)/3).astype(int)
    correct_interpolated_df = correct_interpolated_df.assign(norm_index=real_indices)
    # randomly permute values for split of train, test, and validation
    # split is 70 10 20
    res = correct_interpolated_df.iloc[np.random.permutation(len(correct_interpolated_df))]
    selection = res['norm_index'].values[0]
    
    length = len(res)
    # get splits
    first_split_idx = int(length * 0.7)
    remaining = length - first_split_idx
    second_split_idx = int(first_split_idx + (remaining*0.333))
    train = res.iloc[:first_split_idx]
    valid = res.iloc[first_split_idx:second_split_idx]
    test = res.iloc[second_split_idx:]
    # return split dataframes
    train.to_csv('training_steer.csv')
    valid.to_csv('valid_steer.csv')
    test.to_csv('test_steer.csv')
    return train, valid, test

# rand_split('output')

def move_img(df, data_group, udacity_data_path_folder='output', main_data_file_path='all-data'):
    '''
    moves the image and adds to steering.txt in order.
    '''
    udacity_name = 'UDACITY_{}'.format(data_group.upper())
    arr = np.zeros(len(df))
    for ind,filename in enumerate(df['filename'].values):
        # get initial filename
        fname = os.path.join(udacity_data_path_folder, filename)
        # move to main data folder

        print(os.path.join(main_data_file_path, data_group, udacity_name,filename))
        os.rename(fname, os.path.join(main_data_file_path, data_group, udacity_name, filename))
        # add to file
        angle = df['angle'].values[ind]
        arr[ind] = angle
    steering_fname = 'steering_{}.txt'.format(data_group)
    np.savetxt(steering_fname, arr)
    os.rename(steering_fname, os.path.join(main_data_file_path, data_group, udacity_name, steering_fname))

if __name__ == '__main__':
    train_df, valid_df, test_df = rand_split('output')
    print(train_df.head(), valid_df.head())
    move_img(train_df, 'training')
    move_img(valid_df, 'validation')
    move_img(test_df, 'testing')