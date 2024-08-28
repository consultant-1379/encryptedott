from os import listdir
from os.path import isfile, join, split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np


# Functions for loading / scaling the input features and the target labels

def load_and_scale(folder, data_file_ending, ground_truth_file_ending, split_percentage=0.7, should_filter=False, random_state=42):
    scaler = MinMaxScaler()
    matching_files_of_folder = sorted([f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(data_file_ending)])
    ground_truth_files = sorted([f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(ground_truth_file_ending)])
    loaded_files = load_data(folder, matching_files_of_folder)
    ground_truth_values = [np.loadtxt(join(folder, f)) for f in ground_truth_files]
    if should_filter:
        loaded_files, ground_truth_values = filter_fullzero_rows(loaded_files, ground_truth_values)
    training, testing, training_y, testing_y = train_test_split(loaded_files, ground_truth_values,
                                                                train_size=split_percentage, random_state=random_state)

    scaled_training, scaled_testing = scale_training_and_test_data(training, testing, scaler)
    return scaled_training, training_y, scaled_testing, testing_y


def load_y_values(folder, files):
    loaded_files = []
    for file in files:
        loaded_files.append(np.loadtxt(join(folder,file)))

def filter_fullzero_rows(loaded_files, ground_truth_values):
    filtered_files = []
    filtered_gt = []
    for index, file in enumerate(loaded_files):
        full_zero_row_indices = np.where(~file.any(axis=1))[0]
        consecutive_zero_row_indices = [i for i, next_i in zip(full_zero_row_indices, full_zero_row_indices[1:])
             if next_i - i == 1]
        filtered_files.append(np.delete(file,consecutive_zero_row_indices,0))
        filtered_gt.append(np.delete(ground_truth_values[index], consecutive_zero_row_indices, 0))
    return filtered_files, filtered_gt

def load_data(folder, files):
    loaded_files = []
    for file in files:
        data = np.loadtxt(join(folder, file), delimiter=',', skiprows=1)
        data = np.delete(data, 0, 1)
        data = np.delete(data, 3, 1)
        loaded_files.append(data)
    return loaded_files


def train_test_split_custom(list_of_data, percentage):
    train_index = int(len(list_of_data) * percentage)
    training_data = []
    testing_data = []
    for i, data in enumerate(list_of_data):
        if i < train_index:
            training_data.append(data)
        else:
            testing_data.append(data)
    return training_data, testing_data


def scale_training_and_test_data(training_data, testing_data, scaler):
    for data in training_data:
        scaler.partial_fit(data)
    scaled_training_data = scale_elements_of_list(training_data, scaler)
    scaled_testing_data = scale_elements_of_list(testing_data, scaler)
    return scaled_training_data, scaled_testing_data


def scale_elements_of_list(list_of_data, scaler):
    scaled_data = []
    for data in list_of_data:
        scaled_data.append(scaler.transform(data))
    return scaled_data


if __name__ == '__main__':
    load_and_scale('preprocessed', 'periods.csv', 'gt.csv')
