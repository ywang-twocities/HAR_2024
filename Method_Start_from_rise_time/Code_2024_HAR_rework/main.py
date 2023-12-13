import time

from model_library import model_selection
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

    start = time.time()
    X_train, X_test, y_train, y_test = prepare_data(config, data_dir, exp_envs, scenario) # data preparation
    # model creation, train and evaluation.
    model_func = model_selection('random forest')
    model = model_func(bootstrap=True, max_samples=0.9, max_depth=None, n_estimators=400) #
    print(type(X_train))
    train_and_evaluate_model(X_train, y_train, X_test, y_test, model, config['number_cir'])

    end = time.time()

    print(end-start)
    code.interact(local=locals())


if __name__ == '__main__':
    main()
