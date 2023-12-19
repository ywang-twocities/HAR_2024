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

    elif model_name == 'lstm':
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Permute
        from keras.optimizers import Adam

        def lstm(input_shape, num_classes=7):
            model = Sequential()
            model.add(Permute((2, 1)))  # why adding this layer works so well?
            model.add(LSTM(64, input_shape=input_shape))
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=1e-3),
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
            return model
        return lstm

    elif model_name == 'lstm_fcn':
        from keras import regularizers
        from keras.layers import (Permute, Conv1D, BatchNormalization, GlobalAveragePooling1D, Activation,
                                  concatenate, Input, LSTM, Dense)
        from keras.optimizers import Adam
        from keras.models import Model

        def lstm_fcn(input_shape, num_classes=7):
            inputs = Input(shape=input_shape, name='Input')
            x = Permute((2, 1))(inputs)
            x = LSTM(64)(x)

            y = Conv1D(128, 8, padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv1D(256, 5, padding='same', kernel_regularizer=regularizers.l2(0.01))(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv1D(128, 3, padding='same')(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = GlobalAveragePooling1D()(y)

            x = concatenate([x, y])
            outputs = Dense(num_classes, activation='softmax')(x)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=1e-3),
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
            return model
        return lstm_fcn
