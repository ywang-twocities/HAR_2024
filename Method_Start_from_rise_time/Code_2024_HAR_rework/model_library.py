def model_selection(model_name):
    if model_name == 'random forest':
        from sklearn.ensemble import RandomForestClassifier

        def random_forest(bootstrap=True, max_samples=0.9, max_depth=None, n_estimators=100):
            return RandomForestClassifier(bootstrap=bootstrap, max_samples=max_samples, max_depth=max_depth,
                                          n_estimators=n_estimators)

        return random_forest

    elif model_name == 'svm':
        from sklearn import svm

        def support_vector_machine(kernel='linear', C=1.0, gamma='scale'):
            return svm.SVC(kernel=kernel, C=C, gamma=gamma)

        return support_vector_machine

    elif model_name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier

        def knn(n_neighbors):
            return KNeighborsClassifier(n_neighbors=n_neighbors)

        return knn

    elif model_name == 'naive bayesian':
        from sklearn.naive_bayes import GaussianNB

        def nbc():
            return GaussianNB()

        return nbc

    elif model_name == 'gru':
        from keras.models import Sequential
        from keras.layers import GRU, Dense, Permute
        from keras.optimizers import Adam

        def gru(input_shape, num_classes=7):
            model = Sequential()
            model.add(Permute((2, 1)))  # why adding this layer works so well?
            model.add(GRU(64, input_shape=input_shape))
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=1e-3),
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
            return model

        return gru
