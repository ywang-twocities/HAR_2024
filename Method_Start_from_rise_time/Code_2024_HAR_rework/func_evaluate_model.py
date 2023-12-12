from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, num_cir):
    # training
    model.fit(X_train, y_train)

    # prediction
    predictions = model.predict(X_test)

    # evaluation
    accuracy_train = model.score(X_train, y_train) * 100
    accuracy_test = model.score(X_test, y_test) * 100

    # output the result
    print("Training Accuracy: {:.2f}%".format(accuracy_train))
    print("Test Accuracy: {:.2f}%".format(accuracy_test))
    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, predictions, normalize='true')
    print(cm)
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, predictions))
    print('\n')

    # confusion matrix visualization
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, xticks_rotation='vertical', normalize='true')
    plt.show()

    # 额外的信息
    print('The number of CIR used in model is: {}'.format(X_train.shape[1] / num_cir))
