# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:21:27 2017

@author: RMFit
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:04:22 2017

@author: RMFit
"""
import csv
import cv2
import os.path
import numpy as np

from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU as Lrelu
import matplotlib.pyplot as plt

clear_session()
cut_threshold = 0.2

def image_load(target):
    """ 
    Load and preprocess the target image file.
    
    Args:
        target (str): Path to the image file to be loaded.
    
    Returns:
        2D Float numpy array (64, 64) of the target image.
    """
    image = cv2.imread(target, 0)
    image = image[75:135, :]
    image = cv2.resize(image, (64,64))
    return image

def data_compiler(data_folder):
    """ 
    Collects all of the simulated data and indexes the results.
    
    Args:
        data_folder (str): Path to the parent folder containing all of the simulated datasets.

    Returns:
        proc_log (str): Path to the index file for the compiled image set.
    """
    proc_images = data_folder + '/compiled/'
    src_images = data_folder + '/raw/'
    proc_log = proc_images + 'compiled_samples.csv'
    if not os.path.isdir(proc_images):
        os.mkdir(proc_images)

    image_dirs = os.listdir(src_images)
    image_index = 0
    
    if not os.path.isfile(proc_log): # If the compiled index does not already exist, create it.
        with open(proc_log, 'w', newline='') as compiled:
            writer = csv.DictWriter(compiled, fieldnames={'index', 'reading', 'throttle', 'image'})
            writer.writeheader()
            for directory in image_dirs: # Every directory within '/raw/' is assumed to be simulation data.
                csv_path = src_images + directory + '/driving_log.csv'
                image_folder = src_images + directory + '/IMG/'
                with open(csv_path) as logfile:
                    reader = csv.reader(logfile)
                    for line in reader: # Work through each line of simulated data.
                        reading = float(line[3])
                        throttle = float(line[4])
                        
                        center_cam = image_load(image_folder + line[0].split('\\')[-1])
                        left_cam = image_load(image_folder + line[1].split('\\')[-1])
                        right_cam = image_load(image_folder + line[2].split('\\')[-1])
                        
                        # Perturbs steering reading values and flips images to turn a single sample instance
                        # into 6 data instances.
                        cameras = [(reading, center_cam)]
                        cameras.append((reading-0.2, right_cam))
                        cameras.append((reading+0.2, left_cam))
                        cameras.append((reading*-1, cv2.flip(center_cam, 1)))
                        cameras.append(((reading-0.2)*-1, cv2.flip(right_cam, 1)))
                        cameras.append(((reading+0.2)*-1, cv2.flip(left_cam, 1)))

                        # Work through the list above, writing each image to disk separately.
                        for adj_reading, image in cameras:
                            frame = proc_images + str(image_index) + '_'
                            writer.writerow({'index':image_index, 
                                             'reading':adj_reading,
                                             'throttle':throttle,
                                             'image':frame+'image.png'})
                            cv2.imwrite(frame+'image.png', image)
                            image_index += 1
    return proc_log

def data_filter(index_file, data_folder):
    """ 
    Filters the collected datasets to more evenly present the range of classes.
    
    Args:
        index_file (str): Path to the combined image set index.
        data_folder (str): Path to the image data storage directory.

    Returns:
        filter_log (str): Path to the index file for the filtered image set.
    """
    filter_dir = data_folder + '/filtered/'
    filter_log = filter_dir + 'filtered_samples.csv'
    if not os.path.isdir(filter_dir):
        os.mkdir(filter_dir)
    if not os.path.isfile(filter_log): # If it doesn't already exist, create it.
        samples = dict()
        from collections import defaultdict
        readings = defaultdict(list)
        with open(index_file) as compiled: # Load compiled image index
            reader = csv.DictReader(compiled)
            for s in reader:
                s_index = int(s['index'])
                reading = float(s['reading'])
                throttle = float(s['throttle'])
                image = s['image']
                samples[s_index] = [reading, throttle, image]
                readings[reading].append(s_index)
            counts = defaultdict(list)
            for s in samples: # Split compiled data according to steering angle
                index = samples[s][0]
                index = int(index * 1000)
                counts[index].append(samples[s])
            for c in counts: # Work through each group
                len_c = len(counts[c])
                if abs(c/1000.) <= cut_threshold: # If the data group steering values are close enough to steering 0.
                    sample = np.random.choice(len_c, int(len_c/3)) # Pick 1/3 to keep and ignore the rest.
                    counts[c] = [counts[c][s] for s in sample] # Over-write the group list with the reduced set.
                    
            sample_list = list()
            for c in counts:
                for s in range(len(counts[c])):
                    sample_list.append(counts[c][s]) # Recompile images back into a single list.
            with open(filter_log, 'w', newline='') as logfile:
                writer = csv.DictWriter(logfile, fieldnames={'index', 'reading', 'throttle', 'image'})
                writer.writeheader()
                for i, s in enumerate(sample_list): # Re-index image set
                        frame = filter_dir + str(i) + '_'
                        reading = s[0]
                        throttle = s[1]
                        image = cv2.imread(s[2])
                        cv2.imwrite(frame+'image.png', image)
                        writer.writerow({'index':i,
                                  'reading':reading,
                                  'throttle':throttle,
                                  'image':frame+'image.png'})
    return filter_log

def data_read(index_file):
    """ 
    Read either the compiled or filtered index files and return the contents as a dict.
    
    Args:
        index_file (str): Path to the selected image set index.

    Returns:
        index_dict (dict): Dictionary of data about the images, keys are indexes from file.
    """
    index_dict = dict()
    with open(index_file) as logfile:
        read = csv.DictReader(logfile)
        for line in read:
            index_dict[line['index']] = line
    return index_dict

def data_stats(index_file, low=0, high=6500):
    """ 
    Display a histogram of the steering angles for the dataset.
    
    Args:
        index_file (str): Path to the selected image set index.
        low (int): Lower Y axis threshold.
        high (int): Upper Y axis threshold.
    """
    image_index = data_read(index_file)
    readings = list()
    for row in image_index:
        readings.append(float(image_index[row]['reading']))

    n, bins, patches = plt.hist(readings, 50, facecolor='green', alpha=0.75)    
    plt.xlabel('Reading')
    plt.ylabel('Frequency')
    plt.title('Distribution of steering readings')
    plt.axis([-1, 1, low, high])
    plt.grid(True)
    plt.show()
    
def divide_data(index_file, test_size = 0.2):
    """ 
    Divides the provided data indexes into training, validation and testing sets. 
    The validation and testing sets are the same size, the training set receives 
    the remaining data.
    
    Args:
        index_file (str): Path to the selected image set index.
        test_size (float): Proportion of the dataset to be set aside in testing 
        and validation sets, separately.

    Returns:
        train_set (list): Index data for samples selected for the training set.
        valid_set (list): Index data for samples selected for the validation set.
        test_set (list): Index data for samples selected for the testing set.
    """
    image_dict = data_read(index_file)
    indexes = [x for x in image_dict.keys()]
    test_size = int(len(indexes) * test_size)
    random_indexes = np.random.choice(indexes, size=len(indexes))
    
    data_set = [image_dict[x] for x in random_indexes]
    test_set = data_set[0:test_size]
    valid_set = data_set[test_size:2*test_size]
    train_set = data_set[2*test_size:]
    
    return train_set, valid_set, test_set
        
def make_batch(indexes, batch_size, num_batches):
    """ 
    Generates batches from the provided dataset. 
    
    Args:
        indexes (list): Index data for the dataset provided.
        batch_size (int): Number of samples to provide in each batch.
        num_batches (int): Number of batches to provide in a single epoch.
    Returns:
        x (4D numpy array): Grayscale image data in the form of [batch_size, 64, 64, 1].
        y (1D numpy array): Steering angle ground truth for the matching sample in x.
    """
    x = np.zeros((batch_size, 64, 64, 1), dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.float32)
    while 1:
        for i in range(num_batches):
            for j in range(batch_size):
                x[j,:,:,0] = cv2.imread(indexes[i*batch_size + j]['image'], 0)
                y[j] = float(indexes[i*batch_size + j]['reading']) 
            yield x, y

# Data batching parameters
batch_size = 128
epochs = 40

# Ingest and filter simulator data
datafolder = './data'
compiled_log = data_compiler(datafolder)
balanced_log = data_filter(compiled_log, datafolder)
data_stats(compiled_log, 0, 3000)
data_stats(balanced_log, 0, 500)

# Read index file for filtered data and divide into datasets
data_index = data_read(balanced_log)
train_set, valid_set, test_set = divide_data(balanced_log)
train_batches = len(train_set)//batch_size
valid_batches = len(valid_set)//batch_size
test_batches = len(test_set)//batch_size

# Initialize batching generator for each dataset
train_gen = make_batch(train_set, batch_size, train_batches)
valid_gen = make_batch(valid_set, batch_size, valid_batches)
test_gen = make_batch(test_set, batch_size, test_batches)

# Train model and print summary
model = Sequential()
model.add(Conv2D(12, (3, 3), strides=(2, 2), kernel_initializer="glorot_normal", input_shape=(64, 64, 1)))
model.add(Lrelu(0.3))
model.add(Conv2D(16, (3, 3), strides=(2, 2), kernel_initializer="glorot_normal"))
model.add(Lrelu(0.3))
model.add(Conv2D(24, (3, 3), strides=(2, 2), kernel_initializer="glorot_normal"))
model.add(Lrelu(0.3))

model.add(Flatten())
model.add(Dense(1000, kernel_initializer="glorot_normal"))
model.add(Lrelu(0.3))
model.add(Dense(100, kernel_initializer="glorot_normal"))
model.add(Lrelu(0.3))
model.add(Dense(1, kernel_initializer="glorot_normal"))
model.compile(loss='mse', optimizer='adam')

trained_history = model.fit_generator(train_gen,
                        steps_per_epoch=train_batches, 
                        epochs=epochs,
                        validation_data=valid_gen,
                        validation_steps=valid_batches)
model.save('./model.h5')
print(model.summary())

# Calculate final test loss for trained model
test_model = list()
for sample in test_set:
    image = cv2.imread(sample['image'], 0)
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 3)
    test_model.append((model.predict(image), float(sample['reading'])))
test_result = [(x[0] - x[1]) ** 2 for x in test_model]
print(np.mean(test_result))