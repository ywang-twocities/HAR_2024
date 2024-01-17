import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, num_cir, model_name, label_encoder):
    # training
    if model_name in ['gru', 'lstm', 'lstm_fcn', 'rnn', '2d_fcn', 'cnn', 'mlp']:
        # Normalize X_train, X_test first before reshaping.
        # Accordingly, axis=-1 indicates normalizing along each timestep.
        from keras.layers import Normalization
        norm = Normalization(axis=-1)
        norm.adapt(X_train)
        X_train = norm(X_train)
        X_test = norm(X_test)
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        if model_name in ['gru', 'lstm', 'lstm_fcn', 'rnn', '2d_fcn']:
            # Reshape X_train, X_test into dimensions of (samples, time_steps, feature) for certain models.
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        elif model_name in ['cnn', 'mlp']:
            # Reshape X_train, X_test into dimensions of (samples, feature) for cnn.
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

        y_train = to_categorical(y_train, num_classes=7)
        y_test_encoded = to_categorical(y_test, num_classes=7)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=1e-6)
        history = model.fit(X_train,
                            y_train,
                            epochs=100,
                            validation_data=(X_test, y_test_encoded),
                            batch_size=32,
                            verbose=0,
                            callbacks=[reduce_lr]
                            )
        predictions = model.predict(X_test)
        predictions = np.argmax(predictions, axis=1)
        loss_train, accuracy_train = model.evaluate(X_train, y_train)
        loss_test, accuracy_test = model.evaluate(X_test, y_test_encoded)
        accuracy_train *= 100
        accuracy_test *= 100

    else:
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy_train = model.score(X_train, y_train) * 100
        accuracy_test = model.score(X_test, y_test) * 100

    predictions_str = label_encoder.inverse_transform(predictions)
    y_test_str = label_encoder.inverse_transform(y_test)

    # output the result
    print("Training Accuracy: {:.2f}%".format(accuracy_train))
    print("Test Accuracy: {:.2f}%".format(accuracy_test))
    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_test_str, predictions_str, normalize='true')
    print(np.round(cm, 2))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test_str, predictions_str))
    print('\n')

    # confusion matrix visualization
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, xticks_rotation='vertical', normalize='true')
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

    print('The number of CIR used in model is: {}'.format(X_train.shape[1] / num_cir))

    # Store overall model accuracy
    overall_accuracy = accuracy_test  # or accuracy_train, depending on which you need

    report = classification_report(y_test_str, predictions_str, output_dict=True)
    act_accuracies = {label: report[label]['precision'] for label in report if label in label_encoder.classes_}
    return act_accuracies, overall_accuracy


