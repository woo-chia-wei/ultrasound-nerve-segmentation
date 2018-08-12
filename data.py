import os
import cv2
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

data_path = 'raw/'
image_rows = 420
image_cols = 580

def create_train_and_test_data():
    # Read raw dataset
    print(datetime.datetime.now(), 'Reading all raw files...')
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    images = sorted(images, key=sort_raw_files)

    # Construct dataframe with is_found label
    print(datetime.datetime.now(), 'Identifying images with / without segments...')
    image_data = []

    for image_name in images:
        if 'mask' in image_name:
            continue
            
        name = image_name.split('.')[0]
        image_mask_name = name + '_mask.tif'
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        
        data = {
            'image_name': image_name,
            'mask_image_name': image_mask_name,
            'is_found': img_mask.max() > 0,
            'name': name
        }

        image_data.append(data)
        
    data = pd.DataFrame(image_data)
    data.set_index('name', inplace=True)

    # Stratified Smapling
    print(datetime.datetime.now(), 'Perform stratified sampling into train-test datasets (70%-30%)...')
    X = data
    y = data['is_found']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    X_train_new = X_train.copy()
    X_train_new['type']='train'
    X_train = X_train_new

    X_test_new = X_test.copy()
    X_test_new['type']='test'
    X_test = X_test_new

    # Construct full set of train-test data
    data = pd.concat([X_train, X_test], axis='index')

    # Split into train-set feature-target dataset
    print(datetime.datetime.now(), 'Converting images data into numpy arrays...')
    X_train = create_image_arrays('image_name', 'train', data)
    y_train = create_image_arrays('mask_image_name', 'train', data)
    X_test = create_image_arrays('image_name', 'test', data)
    y_test = create_image_arrays('mask_image_name', 'test', data)

    # Save to local np arrays
    print(datetime.datetime.now(), 'Save all numpy array to local npy files...')
    np.save('imgs_train.npy', X_train)
    np.save('imgs_mask_train.npy', y_train)
    np.save('imgs_test.npy', X_test)
    np.save('imgs_mask_test.npy', y_test)

    print(datetime.datetime.now(), 'The code was executed successfully.')

def create_image_arrays(data_type, learn_type, data):
    filenames = data.query('type=="{}"'.format(learn_type))[data_type].tolist()
    result = np.ndarray((len(filenames), 1, image_rows, image_cols), dtype=np.uint8)
    for index, filename in enumerate(filenames):
        img = cv2.imread(os.path.join(data_path, 'train', filename), cv2.IMREAD_GRAYSCALE)
        result[index] =  np.array([img])
    return result

def sort_raw_files(fname):
    fname = fname.replace('.tif', '')
    fname = fname.replace('_mask', '')
    parts = fname.split('_')
    return int(parts[0]) * 1000 + int(parts[1])

def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_mask_test = np.load('imgs_mask_test.npy')
    return imgs_test, imgs_mask_test

if __name__ == '__main__':
    create_train_and_test_data()
