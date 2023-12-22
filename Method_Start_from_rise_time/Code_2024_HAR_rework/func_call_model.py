from func_evaluate_model import train_and_evaluate_model


def model_call(model_name, model_func, X_train, y_train, X_test, y_test, config, label_encoder):
    if model_name in ['gru', 'lstm', 'lstm_fcn', 'rnn', '2d_fcn']:
        model = model_func(input_shape=(X_train.shape[1], 1), num_classes=7)  # input_shape=(time_steps, feature=1)
        train_and_evaluate_model(X_train, y_train, X_test, y_test, model, config['number_cir'], model_name,
                                 label_encoder)
    elif model_name in ['cnn']:
        model = model_func(input_shape=(X_train.shape[1], 1), f_size=[3, 3], nb_filters=[6, 16], num_classes=7)
        train_and_evaluate_model(X_train, y_train, X_test, y_test, model, config['number_cir'], model_name,
                                 label_encoder)
    elif model_name in ['mlp']:
        model = model_func(input_shape=(X_train.shape[1],), num_classes=7)
        # input_shape=(X_train.shape[1], ) or input_shape=(X_train.shape[1]) both work.
        train_and_evaluate_model(X_train, y_train, X_test, y_test, model, config['number_cir'], model_name,
                                 label_encoder)
    elif model_name in ['svm']:
        model = model_func(kernel='linear', C=1.0, gamma='scale')
        train_and_evaluate_model(X_train, y_train, X_test, y_test, model, config['number_cir'], model_name,
                                 label_encoder)
    elif model_name in ['knn']:
        model = model_func(n_neighbors=3)
        train_and_evaluate_model(X_train, y_train, X_test, y_test, model, config['number_cir'], model_name,
                                 label_encoder)
    elif model_name in ['random forest']:
        model = model_func(bootstrap=True, max_samples=0.9, max_depth=None, n_estimators=100)
        train_and_evaluate_model(X_train, y_train, X_test, y_test, model, config['number_cir'], model_name,
                                 label_encoder)
    elif model_name in ['naive bayesian']:
        model = model_func()
        train_and_evaluate_model(X_train, y_train, X_test, y_test, model, config['number_cir'], model_name,
                                 label_encoder)
