from func_CIR_processing import load_config
from func_prepare_data import prepare_data
from func_evaluate_model import train_and_evaluate_model
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import code


def main():

    current_dir = Path(__file__)
    exp_envs = ["Complex_LOS-23-12-2022", "LOS-23-12-2022", "NLOS-22-12-2022"]
    scenario = 0
    data_dir = current_dir.parents[2] / 'Dataset_12CIR' / exp_envs[scenario]
    config = load_config('config.json')

    # data preparation
    X_train, X_test, y_train, y_test = prepare_data(config, data_dir, exp_envs, scenario)
    # model selection
    model = RandomForestClassifier(bootstrap=True, max_samples=0.9, max_depth=None, n_estimators=400)
    # train/evaluate models
    train_and_evaluate_model(X_train, y_train, X_test, y_test, model, config['number_cir'])

    #
    # '''READ data from data recording folders'''
    # number_cir = config['number_cir']
    # sample_start = config['sample_start']
    # sample_length = config['sample_length']
    # X = []
    # y = []
    # dataset = file_with_label(data_dir)
    # for file, label in dataset:
    #     X.append(read_log_file(file, number_cir, sample_start, sample_length))
    #     y.append(label)
    #
    # '''COMBINE and CROP cir length for further processing'''
    # # crop and combine cir sequences.
    # X_cropped = []
    # length_cir = config['length_cir']
    # nr = config['nr']
    # rate_to_max = config['rate_to_max']
    # for example in X:
    #     example_i, _ = crop_combine(example, length_cir, nr, rate_to_max)
    #     X_cropped.append(example_i)
    # # exclude the damaged data if the length does not correspond to our defined length of number_cir * length_cir.
    # X_cropped, y = zip(*[(element, y[i]) for i, element in enumerate(X_cropped) if len(element) == number_cir * length_cir])
    # X_cropped, y = list(X_cropped), list(y)
    #
    # '''SPLIT'''
    # if exp_envs[scenario].startswith('LOS') or exp_envs[scenario].startswith('Complex'):
    #     X_train, X_test, y_train, y_test = train_test_split(X_cropped, y, test_size=0.2, random_state=40)
    # elif exp_envs[scenario].startswith('NLOS'):
    #     X_train, X_test, y_train, y_test = train_test_split(X_cropped, y, test_size=0.4, random_state=40)
    #     X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=40)

    code.interact(local=locals())


if __name__ == '__main__':
    main()
