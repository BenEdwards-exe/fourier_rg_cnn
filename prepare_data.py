
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import rotate
from sklearn.utils import shuffle

from CRUMB import *

def prepare_all_data(random_seed, isAugment=True, isNormalise=True, isPrint=False):
    # CRUMB Labels are [MiraBest, FR-DEEP, AT17, MiraBest Hybrid]
    test_data = CRUMB('crumb', download=True, train=False, transform=None)
    train_data = CRUMB('crumb', download=True, train=True, transform=None)

    # Isolate labels for FR-DEEP
    fr_deep_train_labels = np.array(train_data.complete_labels)[:, 1]
    fr_deep_test_labels = np.array(test_data.complete_labels)[:, 1]
    # Find FR-Deep Indices in CRUMB
    fr_deep_train_indices = np.flatnonzero(np.isin(fr_deep_train_labels, np.array([0, 1])))
    fr_deep_test_indices = np.flatnonzero(np.isin(fr_deep_test_labels, np.array([0, 1])))

    # Isolate labels for AT17
    at17_train_labels = np.array(train_data.complete_labels)[:, 2]
    at17_test_labels = np.array(test_data.complete_labels)[:, 2]
    # Find AT17 Indices in CRUMB
    at17_train_indices = np.flatnonzero(np.isin(at17_train_labels, np.array([0, 1, 2])))
    at17_test_indices = np.flatnonzero(np.isin(at17_test_labels, np.array([0, 1, 2])))

    # Concatenate Indices and remove duplicates
    temp_concat = np.concatenate((fr_deep_train_indices, at17_train_indices))
    train_indices = np.array(list(set(temp_concat)))
    temp_concat = np.concatenate((fr_deep_test_indices, at17_test_indices))
    test_indices = np.array(list(set(temp_concat)))

    # Isolate Train and Test data
    x_train_lst = []
    y_train_lst = []

    for index in train_indices:
        x_train_lst.append(train_data.data[index])
        # Get the label. AT17 takes precedent over FR-Deep
        label = train_data.complete_labels[index][2]
        if (label == -1):
            label = train_data.complete_labels[index][1]
        y_train_lst.append(label)

    x_test_lst = []
    y_test_lst = []
    for index in test_indices:
        x_test_lst.append(test_data.data[index])
        # Get the label. AT17 takes precedent over FR-Deep
        label = test_data.complete_labels[index][2]
        if (label == -1):
            label = test_data.complete_labels[index][1]
        y_test_lst.append(label)


    # Arrays For Train, Test, Valid
    # Split Train into Train and Valid
    x_test = np.array(x_test_lst)
    y_test = np.array(y_test_lst)
    x_train, x_valid, y_train, y_valid = train_test_split(np.array(x_train_lst), np.array(y_train_lst), train_size=0.8, random_state=random_seed, shuffle=True)


    if isPrint:
        classes = ["FRI", "FRII", "BENT"]

        print("-"*5+" Train Data "+"-"*5)
        count = count_labels(y_train)
        print(f"{classes[0]}:\t{count[0]}\t({count[0]*24})")
        print(f"{classes[1]}:\t{count[1]}\t({count[1]*24})")
        print(f"{classes[2]}:\t{count[2]}\t({count[2]*24})")
        print(f"Total:\t{y_train.shape[0]}\t({y_train.shape[0]*24})")
        print("-"*22)


        print("-"*2+" Validation Data "+"-"*3)
        count = count_labels(y_valid)
        print(f"{classes[0]}:\t{count[0]}\t({count[0]*24})")
        print(f"{classes[1]}:\t{count[1]}\t({count[1]*24})")
        print(f"{classes[2]}:\t{count[2]}\t({count[2]*24})")
        print(f"Total:\t{y_valid.shape[0]}\t({y_valid.shape[0]*24})")
        print("-"*22)

        print("-"*5+" Test Data "+"-"*6)
        count = count_labels(y_test)
        print(f"{classes[0]}:\t{count[0]}\t({count[0]*24})")
        print(f"{classes[1]}:\t{count[1]}\t({count[1]*24})")
        print(f"{classes[2]}:\t{count[2]}\t({count[2]*24})")
        print(f"Total:\t{y_test.shape[0]}\t({y_test.shape[0]*24})")
        print("-"*22)

    # Agument Dataset
    if isAugment:
        x_train, y_train = augment_with_rotations(x_train, y_train, 24)
        x_valid, y_valid = augment_with_rotations(x_valid, y_valid, 24)
        x_test, y_test = augment_with_rotations(x_test, y_test, 24)

    # Normalise Dataset
    if isNormalise:
        mean = np.mean(x_train)
        std = np.std(x_train)

        x_train = (x_train - mean) / (std + 1e-7)
        x_valid = (x_valid - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)

    # Shuffle Dataset
    x_train, y_train = shuffle(x_train, y_train, random_state=random_seed)
    x_valid, y_valid = shuffle(x_valid, y_valid, random_state=random_seed)
    x_test, y_test = shuffle(x_test, y_test, random_state=random_seed)

    return x_train, y_train, x_valid, y_valid, x_test, y_test



def count_labels(y_arr):
    count_arr = np.array([0, 0, 0])
    for label in y_arr:
        prev = count_arr[label]
        prev += 1
        count_arr[label] = prev

    return count_arr

def augment_with_rotations(x, y, n_rotations):
    x_rotate_temp = []
    y_rotate_temp = []
    for i in range(x.shape[0]):
        for step in range(n_rotations):
            x_rotate_temp.append(rotate(image=x[i], angle=step*15, resize=False))
            y_rotate_temp.append(y[i])
    return np.array(x_rotate_temp), np.array(y_rotate_temp)

if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_all_data(random_seed=12, isPrint=True)
