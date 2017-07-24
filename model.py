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
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU as Lrelu
from keras.optimizers import Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt

cut_threshold = 0.2

def image_load(target):
    image = cv2.imread(target, 0)
    image = image[75:135, :]
    image = cv2.resize(image, (64,64))
    return image

def data_compiler(data_folder, force=False):
    proc_images = data_folder + '/compiled/'
    src_images = data_folder + '/raw/'
    proc_log = proc_images + 'compiled_samples.csv'
    if not os.path.isdir(proc_images):
        os.mkdir(proc_images)

    image_dirs = os.listdir(src_images)
    image_index = 0
    
    if force or not os.path.isfile(proc_log):
        with open(proc_log, 'w', newline='') as compiled:
            writer = csv.DictWriter(compiled, fieldnames={'index', 'reading', 'throttle', 'image'})
            writer.writeheader()
            for directory in image_dirs:
                csv_path = src_images + directory + '/driving_log.csv'
                image_folder = src_images + directory + '/IMG/'
                with open(csv_path) as logfile:
                    reader = csv.reader(logfile)
                    for line in reader:
                        reading = float(line[3])
                        throttle = float(line[4])
                        
                        center_cam = image_load(image_folder + line[0].split('\\')[-1])
                        left_cam = image_load(image_folder + line[1].split('\\')[-1])
                        right_cam = image_load(image_folder + line[2].split('\\')[-1])
                        
                        cameras = [(reading, center_cam)]
                        #if abs(reading) > cut_threshold:
                            #reading += np.random.normal(0., 0.1)
                        cameras.append((reading-0.2, right_cam))
                        cameras.append((reading+0.2, left_cam))
                        cameras.append((reading*-1, cv2.flip(center_cam, 1)))
                        cameras.append(((reading-0.2)*-1, cv2.flip(right_cam, 1)))
                        cameras.append(((reading+0.2)*-1, cv2.flip(left_cam, 1)))

                        for adj_reading, image in cameras:
                            frame = proc_images + str(image_index) + '_'
                            writer.writerow({'index':image_index, 
                                             'reading':adj_reading,
                                             'throttle':throttle,
                                             'image':frame+'image.png'})
                            cv2.imwrite(frame+'image.png', image)
                            image_index += 1
    return proc_log

def data_filter(index_file, data_folder, force=False):
    filter_dir = data_folder + '/filtered/'
    filter_log = filter_dir + 'filtered_samples.csv'
    if not os.path.isdir(filter_dir):
        os.mkdir(filter_dir)
    if force or not os.path.isfile(filter_log):
        samples = dict()
        from collections import defaultdict
        readings = defaultdict(list)
        with open(index_file) as compiled:
            reader = csv.DictReader(compiled)
            for s in reader:
                s_index = int(s['index'])
                reading = float(s['reading'])
                throttle = float(s['throttle'])
                image = s['image']
                samples[s_index] = [reading, throttle, image]
                readings[reading].append(s_index)
            counts = defaultdict(list)
            for s in samples:
                index = samples[s][0]
                index = int(index * 1000)
                counts[index].append(samples[s])
            for c in counts:
                len_c = len(counts[c])
                #print(c/1000., '\t', len_c, end='\t')
                
                if abs(c/1000.) <= cut_threshold:
                    sample = np.random.choice(len_c, int(len_c/3))
                    counts[c] = [counts[c][s] for s in sample]
                #elif abs(c/1000.) <= 0.2:
                #    sample = np.random.choice(len_c, int(len_c/6))
                #    counts[c] = [counts[c][s] for s in sample]
                #elif abs(c/1000.) <= 0.25:
                #    sample = np.random.choice(len_c, int(len_c/4))
                #    counts[c] = [counts[c][s] for s in sample]
                #elif abs(c/1000.) <= 0.3:
                #    sample = np.random.choice(len_c, int(len_c/3))
                #    counts[c] = [counts[c][s] for s in sample]
                #print(len(counts[c]))
            #print(sum([len(counts[c]) for c in counts]))
            sample_list = list()
            #print(len(counts))
            for c in counts:
                for s in range(len(counts[c])):
                    sample_list.append(counts[c][s])
            with open(filter_log, 'w', newline='') as logfile:
                writer = csv.DictWriter(logfile, fieldnames={'index', 'reading', 'throttle', 'image'})
                writer.writeheader()
                for i, s in enumerate(sample_list):
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
    index_dict = dict()
    with open(index_file) as logfile:
        read = csv.DictReader(logfile)
        for line in read:
            index_dict[line['index']] = line
    return index_dict

def data_stats(index_file, low=0, high=6500):
    image_index = data_read(index_file)
    readings = list()
    for row in image_index:
        readings.append(float(image_index[row]['reading']))
    
    # the histogram of the data
    n, bins, patches = plt.hist(readings, 50, facecolor='green', alpha=0.75)
    
    plt.xlabel('Reading')
    plt.ylabel('Frequency')
    plt.title('Distribution of steering readings')
    plt.axis([-1, 1, low, high])
    plt.grid(True)
    
    plt.show()
    
def divide_data(index_file, test_size = 0.2):
    image_dict = data_read(index_file)
    indexes = [x for x in image_dict.keys()]
    test_size = int(len(indexes) * test_size)
    random_indexes = np.random.choice(indexes, size=len(indexes))
    data_set = [image_dict[x] for x in random_indexes]

    test_set = data_set[0:test_size]
    valid_set = data_set[test_size:2*test_size]
    train_set = data_set[2*test_size:]
    
    return train_set, valid_set, test_set

def small_batch(images):
    x = np.zeros((5,64,64,1))
    y = np.zeros((5))
    while 1:
        for s, sample in enumerate(images):
            x[s,:,:,0] = sample[1]
            y[s] = sample[0]
        yield x, y
        
def make_batch(indexes, batch_size, num_batches):
    x = np.zeros((batch_size, 64, 64, 1), dtype=np.float32)
    y = np.zeros(batch_size, dtype=np.float32)
    while 1:
        for i in range(num_batches):
            for j in range(batch_size):
                x[j,:,:,0] = cv2.imread(indexes[i*batch_size + j]['image'], 0)
                y[j] = float(indexes[i*batch_size + j]['reading']) 
            yield x, y
    

#def make_batch(index_file, batch_size, image_stable=1., label_stable=1.):
#    image_index = data_read(index_file)
#    indexes = [x for x in image_index.keys()]
#    x = np.zeros((batch_size, 64, 64, 1), dtype=np.float32)
#    y = np.zeros(batch_size, dtype=np.float32)
#    while 1:
#        sample_set = np.random.choice(indexes, size=batch_size)
#        for i, index in enumerate(sample_set):
#            x[i,:,:,0] = cv2.imread(image_index[index]['image'])
#            y[i] = float(image_index[index]['reading']) 
#        yield x, y

datafolder = './data'
compiled_log = data_compiler(datafolder)
balanced_log = data_filter(compiled_log, datafolder)
data_stats(compiled_log, 0, 3000)
data_stats(balanced_log, 0, 500)
data_index = data_read(balanced_log)
train_set, valid_set, test_set = divide_data(balanced_log)
clear_session()
def keras_model():
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
    #model.add(Dropout(keep_prob))

    model.add(Dense(100, kernel_initializer="glorot_normal"))
    model.add(Lrelu(0.3))
    #model.add(Dropout(keep_prob))
    
    model.add(Dense(1, kernel_initializer="glorot_normal"))

    model.compile(loss='mse', optimizer='adam')
    return model

def simple_gen():
    images = [(-0.5, cv2.imread("./data/simple/23011_image.png", 0)), # far_left
              (-0.25, cv2.imread("./data/simple/6_image.png", 0)), # near_left
              (0, cv2.imread("./data/simple/10181_image.png", 0)), # center
              (0.25, cv2.imread("./data/simple/497_image.png", 0)), # near_right
              (0.5, cv2.imread("./data/simple/17281_image.png", 0))]#far_right
    x = np.zeros((5, 64, 64, 1))
    y = np.zeros((5))
    for s, sample in enumerate(images):
        y[s] = sample[0]
        x[s,:,:,0] = sample[1]
    while 1:
        yield x, y

def prove_model():
    model = keras_model()
    gen = simple_gen()
    model.fit_generator(gen, steps_per_epoch=50, epochs=5, validation_data=gen, validation_steps=1)
    for x, y in gen:
        print(y[0], model.predict(x[0:1,:,:,:]))
        print(y[1], model.predict(x[1:2,:,:,:]))
        print(y[2], model.predict(x[2:3,:,:,:]))
        print(y[3], model.predict(x[3:4,:,:,:]))
        print(y[4], model.predict(x[4:,:,:,:]))
        break
    model.save("./simple_model.h5")
#prove_model()


def train_model(model_name, train_gen, valid_gen, train_batches=1, valid_batches=1, epochs=5):
    model = keras_model()
    history = model.fit_generator(train_gen,
                        steps_per_epoch=train_batches, 
                        epochs=epochs,
                        validation_data=valid_gen,
                        validation_steps=valid_batches)
    model.save(model_name)
    return model, history

batch_size = 128
epochs = 40
keep_prob = 0.5

train_batches = len(train_set)//batch_size
valid_batches = len(valid_set)//batch_size
test_batches = len(test_set)//batch_size

train_gen = make_batch(train_set, batch_size, train_batches)
valid_gen = make_batch(valid_set, batch_size, valid_batches)
test_gen = make_batch(test_set, batch_size, test_batches)

trained_model, trained_history = train_model('./model.h5', train_gen, valid_gen, train_batches, valid_batches, epochs)
print(trained_model.summary())

test_model = list()
for sample in test_set:
    image = cv2.imread(sample['image'], 0)
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 3)
    test_model.append((trained_model.predict(image), float(sample['reading'])))
test_result = [(x[0] - x[1]) ** 2 for x in test_model]
       
print(np.mean(test_result))