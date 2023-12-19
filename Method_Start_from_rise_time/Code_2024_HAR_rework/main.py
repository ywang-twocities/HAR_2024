import time

from model_library import model_selection
from sklearn.preprocessing import LabelEncoder
from func_CIR_processing import load_config
from func_prepare_data import prepare_data, split_data
from func_evaluate_model import train_and_evaluate_model
from pathlib import Path
import code


def main():
    current_dir = Path(__file__)
    exp_envs = ["Complex_LOS-23-12-2022", "LOS-23-12-2022", "NLOS-22-12-2022"]
    scenario = 1
    data_dir = current_dir.parents[2] / 'Dataset_12CIR' / exp_envs[scenario]
    config = load_config('config.json')

    start = time.time()

    X, y = prepare_data(config, data_dir)  # data preparation
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = split_data(X, y_encoded, exp_envs, scenario)

    # model creation, train and evaluation.
    model_name = 'lstm_fcn'

    if model_name in ['gru', 'lstm', 'lstm_fcn', 'rnn', '2D_fcn', 'mlp']:
        model_func = model_selection(model_name)
        print(X_train.shape)
        model = model_func(input_shape=(X_train.shape[1], 1), num_classes=7)  # input_shape=(time_steps, feature=1)
        train_and_evaluate_model(X_train, y_train, X_test, y_test, model, config['number_cir'], model_name, label_encoder)

    end = time.time()
    print(end - start)

    code.interact(local=locals())


if __name__ == '__main__':
    main()
