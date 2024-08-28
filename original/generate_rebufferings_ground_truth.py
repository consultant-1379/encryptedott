from os import listdir, makedirs
from os.path import isfile, join, split, exists
from shutil import copyfile
import numpy as np

INPUT_DIR = 'tsr_nf/buffering'
OUTPUT_DIR = 'preprocessed_nf'


def is_element_in_list_of_lists(element, nested_list):
    for sublist in nested_list:
        if element in range(sublist[0], int(sublist[1])):
            return True
    return False

def get_buffering_intervals_from_reference_seattle_format(reference_file, offset):
    buffering_intervals = []
    with open(reference_file, 'r') as ref_file:
        for line in ref_file:
            if line.startswith("##OFFSET"):
                line_parts = line.split(":")
                offset = int(line_parts[1])
            elif line.startswith('#') and len(line.split('-')) == 2:
                line_parts = line.split('-')
                minute_and_sec = line_parts[0]
                time_parts = minute_and_sec.split(':')
                seconds = int(time_parts[0][1:]) * 60 + int(time_parts[1]) - offset
                duration = np.ceil(float(line_parts[1].split(' ')[1].rstrip()))
                # print(seconds, duration)
                buffering_intervals.append((seconds, seconds+duration))
    return buffering_intervals

def get_buffering_intervals_from_reference_hong_kong_format(reference_file, offset=1):
    buffering_intervals = []
    with open(reference_file, 'r', encoding='utf-8') as ref_file:
        for line in ref_file:
            if line.startswith("##OFFSET"):
                line_parts = line.split(":")
                offset = int(line_parts[1])
            if line.startswith("##BUFFERING"):
                line_parts = line.split(":")
                buffering_from = int(line_parts[1].split('-')[0]) - offset
                buffering_to = int(line_parts[1].split('-')[1])
                buffering_intervals.append((buffering_from, buffering_to))
    return buffering_intervals

def generate_ground_truth(orig_filename, offset=23, output_dir='preprocessed'):
    only_filename = split(orig_filename)[-1]
    periods_file_name = only_filename[:-4] + '_periods.csv'
    periods_file = join(output_dir, periods_file_name)
    reference_file = orig_filename[:-4] + '_reference'
    buffering_intervals = []
    if 'seattle' in reference_file or 'android_hu' in reference_file:
        buffering_intervals = get_buffering_intervals_from_reference_seattle_format(reference_file, offset)
    if 'hong_kong' in reference_file or 'edward' in reference_file or 'fb_hu' in reference_file:
        buffering_intervals = get_buffering_intervals_from_reference_hong_kong_format(reference_file)

    if len(buffering_intervals) > 0:
        output_dir = join(output_dir, 'buffering')
    else:
        output_dir = join(output_dir, 'no_buffering')
    if not exists(output_dir):
        makedirs(output_dir)
    copyfile(periods_file, join(output_dir, periods_file_name))
    output_filename = join(output_dir, only_filename[:-4] + '_gt.csv')
    with open(periods_file, 'r', encoding='utf-8') as input_file, open(output_filename, 'w', encoding='utf-8') as output_file:
        input_lines = input_file.readlines()
        for i in range(len(input_lines)-1):
            if is_element_in_list_of_lists(i, buffering_intervals):
                output_file.write(str(1) + '\n')
            else:
                output_file.write(str(0) + '\n')


if __name__ == '__main__':
    onlyfiles = sorted([f for f in listdir(INPUT_DIR) if isfile(join(INPUT_DIR, f)) and f.endswith('.tsr')])
    for f in onlyfiles:
        generate_ground_truth(join(INPUT_DIR, f), output_dir=OUTPUT_DIR)