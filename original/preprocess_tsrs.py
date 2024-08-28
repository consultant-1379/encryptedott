import json
import csv
from os import listdir
from os.path import isfile, join, split

INPUT_DIR = 'tsr_nf/buffering'
OUTPUT_DIR = 'preprocessed_nf'

metrics = ['active_time-DATA_TRANSFER-UP',
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

columns = ['start',
            'active_connections',
            'bytes_down',
            'bytes_up',
            'duration'] \
          + metrics



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


def extract_periods(filename, output_dir=OUTPUT_DIR):
    output_path = split(filename)[-1][:-4]
    with open(filename, 'r', encoding='utf-8') as inputfile, open(join(output_dir, output_path) + '_periods.csv', 'w', encoding='utf-8', newline='') as outputfile:
        matrix = []
        csvwriter = csv.writer(outputfile)
        csvwriter.writerow(columns)
        for tsr_line in inputfile:
            tsr_line = json.loads(tsr_line)
            tsr_line = tsr_line['record']
            periods = tsr_line['periods']
            for period in periods:
                csv_line = []
                csv_line.append(period['start'])
                csv_line.append(period['active_connections'])
                csv_line.append(period.get('bytes_down', 0))
                csv_line.append(period.get('bytes_up',0))
                csv_line.append(period['dur'])
                for metric in metrics:
                    csv_line.append(get_value_from_json(period, metric))
                if len(matrix) >= 1:
                    number_of_empty_durations_between_periods = ((period['start'] - matrix[-1][0]) / period['dur'])

                    if number_of_empty_durations_between_periods > 1:
                        for i in range(1, int(number_of_empty_durations_between_periods)):
                            csvwriter.writerow([matrix[-1][0] + i * period['dur'], 0, 0, 0, period['dur'], *[0 for i in range(len(metrics))]])

                matrix.append(csv_line)
                csvwriter.writerow(csv_line)




if __name__ == '__main__':
    onlyfiles = [f for f in listdir(INPUT_DIR) if isfile(join(INPUT_DIR, f)) and f.endswith('.tsr')]
    for f in onlyfiles:
        extract_periods(join(INPUT_DIR, f), OUTPUT_DIR)
