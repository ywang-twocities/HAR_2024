import numpy as np
from sklearn.model_selection import train_test_split
from func_CIR_processing import read_log_file, file_with_label, crop_combine


def prepare_data(config, data_dir, exp_envs, scenario):
    """
    prepare data for further model training and testing.

    :param config: config.json
    :param data_dir: data directory
    :param exp_envs: list of different scenarios.
    :param scenario: selected scenario (in main func)
    :return: training and testing sets.
    """
    # read params from config.json
    number_cir = config['number_cir']
    sample_start = config['sample_start']
    sample_length = config['sample_length']
    length_cir = config['length_cir']
    nr = config['nr']
    rate_to_max = config['rate_to_max']

    # load data
    X = []
    y = []
    dataset = file_with_label(data_dir)
    for file, label in dataset:
        X.append(read_log_file(file, number_cir, sample_start, sample_length))
        y.append(label)

    # crop and combine.
    X_cropped = []
    for example in X:
        example_i, _ = crop_combine(example, length_cir, nr, rate_to_max)
        X_cropped.append(example_i)

    # keep only the samples with desired length of number_cir * length_cir
    X_cropped, y = zip(*[(element, y[i]) for i, element in enumerate(X_cropped) if len(element) == number_cir * length_cir])
    X_cropped, y = list(X_cropped), list(y)

    # data transform
    X_train = np.array(X_cropped)
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = np.array(y)

    if exp_envs[scenario].startswith('LOS') or exp_envs[scenario].startswith('Complex'):
        X_train, X_test, y_train, y_test = train_test_split(X_cropped, y, test_size=0.2, random_state=40)
    elif exp_envs[scenario].startswith('NLOS'):
        X_train, X_test, y_train, y_test = train_test_split(X_cropped, y, test_size=0.4, random_state=40)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=40)

    return X_train, X_test, y_train, y_test
