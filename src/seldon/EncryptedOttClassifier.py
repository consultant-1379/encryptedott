from keras.models import load_model
import numpy as np
import json
import pickle

WINDOW_SIZE = 30
START = 30
STRIDE = 1

METRICS = ['active_time-DATA_TRANSFER-UP',
                'active_time-DATA_TRANSFER-DOWN',
                'active_time-CONNECTION-SETUP',
                'active_time-ZERO_WINDOW_WAIT',
                'max_delay-DATA_TRANSFER-UP',
                'max_delay-DATA_TRANSFER-DOWN',
                'max_delay-CONNECTION_SETUP',
                'max_delay-ZERO_WINDOW_WAIT',
                'events-NEW_CONNECTION_INITIATED',
                'events-NEW_CONNECTION_ESTABLISHED',
                'events-CONNECTION_TIMEOUT',
                'events-CONNECTION_END',
                'events-CONNECTION_RESET',
                'events-NEW_PDN_SESSION']
				
def get_value_from_json(period_object, name):
    parts = name.split('-')
    event_num = 0
    for json_object in period_object.get(parts[0], []):
        if parts[0] in ['active_time', 'max_delay']:
            if len(parts) == 3:
                if json_object['type'] == parts[1] and json_object['direction'] == parts[2]:
                    return json_object['value']
            else:
                if json_object['type'] == parts[1]:
                    return json_object['value']
        else:
            if json_object['type'] == parts[1]:
                event_num += 1
    return event_num
	
def extract_periods(json_data):
    data = json.loads(json_data)
    data = data['record']
    periods = data['periods']
    list_of_preprocessed_data = []
    for period in periods:
        preprocessed_data = []
        preprocessed_data.append(period['start'])
        preprocessed_data.append(period['active_connections'])
        preprocessed_data.append(period.get('bytes_down', 0))
        preprocessed_data.append(period.get('bytes_up',0))
        preprocessed_data.append(period['dur'])
        for metric in METRICS:
            preprocessed_data.append(get_value_from_json(period, metric))
        list_of_preprocessed_data.append(preprocessed_data)
    return list_of_preprocessed_data

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

def scale(data, scaler):
    scaled_data = scaler.transform(data)
    return scaled_data
	
class EncryptedOttClassifier(object):

        def __init__(self):
                """ You can load your pre-trained model in here. The instance will be created once when the docker container starts running on the cluster. """
                f = open('scaler.pckl', 'rb')
                self.SCALER = pickle.load(f)
                f.close()
                self.MODEL = load_model('model.h5')
                self.MODEL.load_weights('weights.hdf5')

        def predict(self,preprocessed_data,feature_names=['good connection','bad connection']):
                preprocessed_data = np.delete(preprocessed_data, 0, 1)
                preprocessed_data = np.delete(preprocessed_data, 3, 1)
                scaled_data = scale(preprocessed_data, self.SCALER)
                scaled_data = np.array(scaled_data)
                transformed_data = transform_to_sliding_window(scaled_data, stride=STRIDE, window_size=WINDOW_SIZE, start=START)
                predictions = self.MODEL.predict(np.array(transformed_data))
                return predictions