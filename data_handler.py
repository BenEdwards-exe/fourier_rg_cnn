import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import rotate
from sklearn.utils import shuffle

from CRUMB import *




def isolate_at17_frdeep(data: CRUMB):
    at17_indices = np.where(np.transpose(data.complete_labels)[2] != -1)[0]
    frdeep_indices = np.where(np.transpose(data.complete_labels)[1] != -1)[0]

    # Isolate the unique indices of images in frdeep (i.e., only in frdeep)
    only_in_frdeep_indices = []
    for index in frdeep_indices:
        if index not in at17_indices:
            only_in_frdeep_indices.append(index)
    only_in_frdeep_indices = np.array(only_in_frdeep_indices)    

    isolated_images = []
    isolated_labels = []

    # Isolate images from at17
    for index in at17_indices:
        isolated_images.append(data.data[index])
        isolated_labels.append(data.complete_labels[index][2])

    # Isolate images only occuring in frdeep
    for index in only_in_frdeep_indices:
        isolated_images.append(data.data[index])
        isolated_labels.append(data.complete_labels[index][1])

    return np.array(isolated_images), np.array(isolated_labels)



def print_report(lst):
    classes = ['FRI', 'FRII', 'BENT']
    classes_count = [0, 0, 0]
    for element in lst:
        classes_count[element] = classes_count[element] + 1
    print("----------------------------")
    for i in range(3):
        out = classes[i] + ': ' + str(classes_count[i])
        print(out)
    print("----------------------------")


def rotate_data(x, y, n_rotations):
    x_rot_temp = []
    y_rot_temp = []
    for index in range(x.shape[0]):
        for step in range(n_rotations):
            x_rot_temp.append(rotate(image=x[index], angle=step*15, resize=False))
            y_rot_temp.append(y[index])
    return np.array(x_rot_temp), np.array(y_rot_temp)

# 2-D FFT data array of n_channels
# Return separate arrays for real and imaginary 
def fft_data(input_data):

    [n_input, height, width, n_channels] = input_data.shape

    fft_real = np.zeros_like(input_data, dtype=np.float32)
    fft_imag = np.zeros_like(input_data, dtype=np.float32)

    for i in range(n_input):
        matrix = input_data[i, :, :, :]
        real_channel_temp = np.zeros_like(matrix, dtype=np.float32)
        imag_channel_temp = np.zeros_like(matrix, dtype=np.float32)
        for j in range(n_channels):
            fft2d = np.fft.fft2(matrix[:, :, j])
            real_channel_temp[:, :, j] = fft2d.real
            imag_channel_temp[:, :, j] = fft2d.imag
        fft_real[i, :, :, :] = real_channel_temp
        fft_imag[i, :, :, :] = imag_channel_temp

    return fft_real, fft_imag 


# Download dataset from CRUMB
# Isolate AT17 and FR-DEEP sources and combine
# For each random seed:
#   - Split, normalize, augment and save
#   - FFT augmented sets and save
def prepare_local_copy(n_augmentations=24):

    # Load train and test data from CRUMB
    test_data = CRUMB('crumb', download=True, train=False, transform=None)
    train_data = CRUMB('crumb', download=True, train=True, transform=None)

    # Isolate train and test data only occuring in AT17 and FR-DEEP.
    # For duplicates, AT17 takes priority
    x_train, y_train = isolate_at17_frdeep(train_data)
    x_test, y_test = isolate_at17_frdeep(test_data)

    # Scale and cast to float
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Save isolated training and testing data:
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir('data/original'):
        os.mkdir('data/original')


    np.save('data/original/x_train.npy', x_train)
    np.save('data/original/y_train.npy', y_train)
    np.save('data/original/x_test.npy', x_test)
    np.save('data/original/y_test.npy', y_test)

    # List of random seeds used
    list_of_seeds = [
        1327, 5208, 8208, 3515, 2710,
        4212, 1803, 3226, 2712, 7103,
        4310, 2808, 5916, 7600, 1522
    ]

    if not os.path.isdir('data/augmented'):
        os.mkdir('data/augmented') # Directory to store augmented data

    # TODO: describe function
    for random_seed in list_of_seeds:
        np.random.seed(random_seed)

        # Split train set into training and testing sets
        x_train_split, x_valid_split, y_train_split, y_valid_split = train_test_split(x_train, y_train, train_size=0.80, random_state=random_seed, shuffle=True)

        # Augment (add rotations)
        n_augmentations = 24
        x_train_split_rotated, y_train_split_rotated = rotate_data(x_train_split, y_train_split, n_augmentations)
        x_valid_split_rotated, y_valid_split_rotated = rotate_data(x_valid_split, y_valid_split, n_augmentations)
        x_test_rotated, y_test_rotated = rotate_data(x_test, y_test, n_augmentations)
        # Shuffle Augmentations
        x_train_split_rotated, y_train_split_rotated = shuffle(x_train_split_rotated, y_train_split_rotated, random_state=random_seed)
        x_valid_split_rotated, y_valid_split_rotated = shuffle(x_valid_split_rotated, y_valid_split_rotated, random_state=random_seed)
        x_test_rotated, y_test_rotated = shuffle(x_test_rotated, y_test_rotated, random_state=random_seed)


        # Check if scale between 0 and 1
        if (np.max(x_train) > 1.0):
            print("DATASET NOT SCALED")
        
        # Get mean and std of augmented training set
        mean = np.mean(x_train_split_rotated)
        std = np.std(x_train_split_rotated)

        # Normalize data
        x_train_split_rotated = (x_train_split_rotated - mean) / (std + 1e-7)
        x_valid_split_rotated = (x_valid_split_rotated - mean) / (std + 1e-7)
        x_test_rotated = (x_test_rotated - mean) / (std + 1e-7)


        # Save the split, agumented, normalized, and shuffled data
        new_dir = 'data/augmented/' + str(random_seed)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)

        np.save('data/augmented/'+str(random_seed)+'/x_train.npy', x_train_split_rotated)
        np.save('data/augmented/'+str(random_seed)+'/y_train.npy', y_train_split_rotated)
        
        np.save('data/augmented/'+str(random_seed)+'/x_valid.npy', x_valid_split_rotated)
        np.save('data/augmented/'+str(random_seed)+'/y_valid.npy', y_valid_split_rotated)
        
        np.save('data/augmented/'+str(random_seed)+'/x_test.npy', x_test_rotated)
        np.save('data/augmented/'+str(random_seed)+'/y_test.npy', y_test_rotated)


        ## --- FFT the data --- ##
        x_train_fft_real, x_train_fft_imag = fft_data(x_train_split_rotated)
        x_valid_fft_real, x_valid_fft_imag = fft_data(x_valid_split_rotated)
        x_test_fft_real, x_test_fft_imag = fft_data(x_test_rotated)


        # Save fft data
        np.save('data/augmented/' + str(random_seed) + '/x_train_fft_real.npy', x_train_fft_real)
        np.save('data/augmented/' + str(random_seed) + '/x_train_fft_imag.npy', x_train_fft_imag)

        np.save('data/augmented/' + str(random_seed) + '/x_valid_fft_real.npy', x_valid_fft_real)
        np.save('data/augmented/' + str(random_seed) + '/x_valid_fft_imag.npy', x_valid_fft_imag)

        np.save('data/augmented/' + str(random_seed) + '/x_test_fft_real.npy', x_test_fft_real)
        np.save('data/augmented/' + str(random_seed) + '/x_test_fft_imag.npy', x_test_fft_imag)

        # Print finished
        print("Finished augmentation, fft, and save for seed: ", str(random_seed))


if __name__ == "__main__":
    
    prepare_local_copy(n_augmentations=24)