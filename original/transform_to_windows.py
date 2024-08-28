import numpy as np


def transform_to_discrete_window_with_padding(data_to_convert, window_size=5):
    modulo = data_to_convert.shape[0] % window_size - 1
    if modulo != 0:
        padding = np.zeros((modulo,data_to_convert.shape[1]), dtype=data_to_convert.dtype)
        data_to_convert = np.concatenate((padding, data_to_convert), axis=0)
    number_of_windows = int(data_to_convert.shape[0] / window_size)
    return data_to_convert.reshape(number_of_windows, window_size, data_to_convert.shape[1])


# reshape input to be 3D [samples, timesteps, features]
def transform_to_sliding_window(data_to_convert, stride=1, window_size=30, start=0):
    transformed_data = []
    for i in range(start, data_to_convert.shape[0], stride):
        if len(data_to_convert.shape) > 1:
            timestep_window = np.pad(data_to_convert, ((window_size-1, 0), (0, 0)), mode='constant', constant_values=0)
        else:
            timestep_window = np.pad(data_to_convert, (window_size - 1, 0), mode='constant',
                                     constant_values=0)
        timestep_window = timestep_window[i:i+window_size]
        transformed_data.append(timestep_window)
    return np.array(transformed_data)


def transform_labels(data_to_convert, stride=1, start=0, seek_forward=1):
    reduced_labels = []
    for index in range(start, len(data_to_convert), stride):
        reduced_labels.append(max(data_to_convert[index:index+seek_forward]))
        # reduced_labels.append(median_high(data_to_convert[index:index + stride]))
    return reduced_labels
