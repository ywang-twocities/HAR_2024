import time

from model_library import model_selection
from sklearn.preprocessing import LabelEncoder
from func_CIR_processing import load_config
from func_prepare_data import prepare_data, split_data
from func_call_model import model_call
from pathlib import Path
import code


def main():
    current_dir = Path(__file__)
    exp_envs = ["Complex_LOS-23-12-2022", "LOS-23-12-2022", "NLOS-22-12-2022"]
    scenario = [1]
    # data_dir = current_dir.parents[2] / 'Dataset_12CIR' / exp_envs[scenario]
    config = load_config('config.json')
    number_cir_values = [12]
    length_cir_values = [150]
    model_names = ['knn']
    # number_cir_values = [8, 10, 12]
    # length_cir_values = [50, 75, 100, 125, 150, 175, 200, 225]
    # model_names = ['gru', 'lstm', 'lstm_fcn', 'rnn'50, , '2d_fcn', 'cnn', 'mlp', 'svm', 'knn', 'random forest',
    #                                                                          'naive bayesian']
    results = {}

    start = time.time()
    for sce in scenario:
        data_dir = current_dir.parents[2] / 'Dataset_12CIR' / exp_envs[sce]
        for model_name in model_names:
            for number_cir in number_cir_values:
                for length_cir in length_cir_values:
                    X, y = prepare_data(config, data_dir, number_cir, length_cir)
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)
                    X_train, X_test, y_train, y_test = split_data(X, y_encoded, exp_envs, sce)

                    model_func = model_selection(model_name)
                    accuracies, overall_accuracy = model_call(model_name, model_func, X_train, y_train, X_test, y_test,
                                                              label_encoder, number_cir)
                    result_key = (model_name, number_cir, length_cir)
                    results[result_key] = accuracies
                    with open('experiment_results.txt', 'a') as file:
                        file.write(
                            f"Scenario: {exp_envs[sce]}, Model: {result_key[0]}, Number_CIR: {result_key[1]}, "
                            f"Length_CIR: {result_key[2]}, "
                            f"Accuracies: {accuracies}, Overall Accuracy: {overall_accuracy}%\n")
                    return X, y
    end = time.time()
    print(f"{end - start}\n")


if __name__ == '__main__':
    X, y = main()
    import code
    code.interact(local=dict(globals(), **locals()))
