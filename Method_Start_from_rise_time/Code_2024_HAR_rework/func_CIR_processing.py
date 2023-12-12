import math
import json


def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)


def file_with_label(data_dir):
    """
    The function aims at generating labeled dataset.
    :param data_dir: Data_dir is the Path object for the folder of certain scenario (LOS, NLOS, complex LOS)
    :return: list of tuples (data, label)
    """
    log_files_with_labels = []

    for file in data_dir.glob('**/*.log'):
        label = file.parent.name
        log_files_with_labels.append((file, label))
    return log_files_with_labels


# optimized
def read_log_file(file_name, nb_CIR, sample_start, sample_length):
    """
    :param file_name: log file exported by EVK1000
    :param nb_CIR: number of CIRs we want to use in each file. E.g., There may be 30 CIR with length of 1016 samples
    in a recording, while we want only 15 instead of 30, thus making nb_CIR = 15
    :param sample_start: index of sample from which we want to use for feature extraction among 1016.
    :param sample_length: Length of CIR sample to be used in training.
    :return:
    """
    CIR = []
    chunk = []

    with open(file_name, "r") as log_file:
        parsed_log_lines = [line.split() for line in log_file]

    # Find the starting location of each CIR dataset in a file
    index_CIR_start = [i for i, value in enumerate(parsed_log_lines) if value == ['Accum', 'Len', '1016']]

    # Process each CIR dataset
    for start_index in index_CIR_start[:nb_CIR]:  # Limit the number of CIR datasets processed
        # Extract and process the CIR data
        for line in parsed_log_lines[start_index + 1:start_index + 1017]:
            cir_values = [int(x.replace(',', '')) for x in line]
            # Assume each line has 2 values representing a complex number
            cir_val = math.sqrt(cir_values[0] ** 2 + cir_values[1] ** 2)
            chunk.append(cir_val)

        # Extract the relevant section of the CIR data
        if chunk:
            CIR.append(chunk[sample_start:sample_start + sample_length])
            chunk = []

    return CIR


# optimized
def crop_combine(example, length_cir, nr, rate_to_max):
    """
    This function aims to crop each CIR example for saving only significant values and combine them together for each example.
    Source: Using the Power Delay Profile to Accelerate the Training of Neural Network-Based Classifiers for the Identification of LOS and NLOS UWB Propagation Conditions
    :param example: One example (of length nb_cir(e.g. 21 cirs in an example)*length_cir(1016 samples))
    :param length_cir: number of significant cir samples to be cropped and used
    :param nr: rise time from its actual start to the detected index nh. set empirically (usually 3 is enough)
    :param rate_to_max: beta in the paper to present the ratio with the real peak, beta=0.4 empirically。
    :return:
    """
    crop = []
    nh = []

    # calculate the max*rate value for each cir sequence within a *.log set.
    max_values = [max(cir) * rate_to_max for cir in example]

    for j, cir in enumerate(example):
        for idx, value in enumerate(cir):
            if value >= max_values[j]:
                nh_index = idx
                break
        else:
            continue

        nh.append(nh_index)
        # Ensure the indexes are within reasonable range.
        start = max(0, nh_index - nr)
        end = min(start + length_cir, len(cir))
        crop.extend(cir[start:end])
    return crop, nh

