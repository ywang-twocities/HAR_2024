def model_selection(model_name):
    if model_name == 'random forest':
        from sklearn.ensemble import RandomForestClassifier

        def random_forest(bootstrap, max_samples, max_depth, n_estimators):
            return RandomForestClassifier(bootstrap=bootstrap, max_samples=max_samples, max_depth=max_depth,
                                          n_estimators=n_estimators)

        return random_forest
