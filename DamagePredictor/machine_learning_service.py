import joblib
import os

import constants


def predict_classifications(test_values):
    """
    The predict classifications method to predict damage grade classifications.

    :param test_values: The test values to predict on.
    :return: The prediction classifications.
    """
    import pandas as pd

    # Load trained models for prediction
    dirname = os.path.dirname(__file__)

    path = os.path.join(dirname, constants.MODEL_DIRECTORY)

    pathExists = os.path.exists(path)

    if pathExists:
        le_path = os.path.join(path, constants.LE_MODEL_FILE_NAME)
        scaler_path = os.path.join(path, constants.SCALER_MODEL_FILE_NAME)
        xgb_path = os.path.join(path, constants.XGB_MODEL_FILE_NAME)
        knn_bag_path = os.path.join(path, constants.KNN_BAG_MODEL_FILE_NAME)
        ada_path = os.path.join(path, constants.ADA_MODEL_FILE_NAME)

        le = joblib.load(le_path)
        scaler = joblib.load(scaler_path)
        xgb = joblib.load(xgb_path)
        knn_bag = joblib.load(knn_bag_path)
        ada = joblib.load(ada_path)

        # Format and normalize test dataset
        X_test = pd.read_csv(test_values)
        X_test = X_test.drop("building_id", axis=1)
        X_test = pd.get_dummies(X_test)
        X_test = scaler.transform(X_test)

        # Make predictions using base classifier models
        xgb_predictions = xgb.predict_proba(X_test)
        knn_predictions = knn_bag.predict_proba(X_test)

        # Convert predictions into dataframes
        xgb_test_df = pd.DataFrame(data=xgb_predictions)
        knn_test_df = pd.DataFrame(data=knn_predictions)

        # Create stacked dataset
        S_test = pd.concat([xgb_test_df, knn_test_df], axis=1)

        # Predict using meta classifier model
        predictions = ada.predict(S_test)

        predictions = le.inverse_transform(predictions)

        return predictions
    else:
        raise Exception("No models found")


def train_model(training_values, training_classifiers):
    """
    The train model method to train a classifier model.

    :param training_values: The training values.
    :param training_classifiers: The training classifiers.
    :return: void.
    """

    import pandas as pd
    import joblib

    from xgboost.sklearn import XGBClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import LabelEncoder

    # Check if model directory exists and if not, create it
    dirname = os.path.dirname(__file__)

    path = os.path.join(dirname, constants.MODEL_DIRECTORY)

    pathExists = os.path.exists(path)

    if not pathExists:
        os.makedirs(path)

    # Set absolute paths for all saved models
    le_path = os.path.join(path, constants.LE_MODEL_FILE_NAME)
    scaler_path = os.path.join(path, constants.SCALER_MODEL_FILE_NAME)
    xgb_path = os.path.join(path, constants.XGB_MODEL_FILE_NAME)
    knn_bag_path = os.path.join(path, constants.KNN_BAG_MODEL_FILE_NAME)
    ada_path = os.path.join(path, constants.ADA_MODEL_FILE_NAME)

    # Retrieve, manipulate, and format initial training data
    X_train = pd.read_csv(training_values)
    y_train = pd.read_csv(training_classifiers)

    X_train = X_train.drop("building_id", axis=1)
    y_train = y_train.drop("building_id", axis=1)

    X_train = pd.get_dummies(X_train)

    # Determine and remove outliers
    iso = IsolationForest(n_estimators=200, random_state=42, warm_start=True, contamination=0.01, n_jobs=-1)
    iso.fit(X_train)

    y_predictions_train = iso.predict(X_train)
    y_predictions_train = y_predictions_train.tolist()

    X_train = pd.DataFrame(data=X_train)
    y_train = pd.DataFrame(data=y_train)

    X_train['Outlier'] = y_predictions_train
    y_train['Outlier'] = y_predictions_train

    X_train = X_train[X_train.Outlier != -1]
    y_train = y_train[y_train.Outlier != -1]

    X_train = X_train.drop("Outlier", axis=1)
    y_train = y_train.drop("Outlier", axis=1)

    # Normalize data
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    # Pickle the scaler model to use for normalization of prediction data
    joblib.dump(scaler, scaler_path)

    # Define, train, and pickle label encoder model
    le = LabelEncoder()

    le = le.fit(y_train)

    joblib.dump(le, le_path)

    # Normalize labels
    y_train = le.transform(y_train)

    # Split training data for testing and validation
    X = X_train
    y = y_train
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Define, train, and pickle the gradient boosting decision tree base classifier model
    xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                        colsample_bynode=1, colsample_bytree=0.55, gamma=0.37,
                        learning_rate=0.1, max_delta_step=0, max_depth=13,
                        min_child_weight=0.1, n_estimators=324, n_jobs=-1,
                        nthread=8, num_classes=3, objective='multi:softprob',
                        random_state=0, reg_alpha=3, reg_lambda=1, scale_pos_weight=1,
                        seed=42, silent=None, subsample=0.9, verbosity=1)

    xgb = xgb.fit(X_train, y_train)

    joblib.dump(xgb, xgb_path)

    # Define and train the K-Nearest Neighbors base classifier model
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=2, metric='minkowski',
                               metric_params=None, n_jobs=-1, n_neighbors=8, p=1,
                               weights='uniform')

    knn = knn.fit(X_train, y_train)

    # Define, train, and pickle the bagging ensemble of KNN base classifier models
    knn_bag = BaggingClassifier(base_estimator=knn,
                                bootstrap=False, bootstrap_features=False, max_features=0.5,
                                max_samples=1.0, n_estimators=15, n_jobs=-1, oob_score=False,
                                verbose=0, warm_start=False)

    knn_bag = knn_bag.fit(X_train, y_train)

    joblib.dump(knn_bag, knn_bag_path)

    # Generate cross-validated estimates
    xgb_probs = cross_val_predict(xgb, X_train, y_train, method='predict_proba', cv=5, n_jobs=-1)
    knn_probs = cross_val_predict(knn_bag, X_train, y_train, method='predict_proba', cv=5, n_jobs=-1)

    # Convert predictions into dataframes
    xgb_df = pd.DataFrame(data=xgb_probs)
    knn_df = pd.DataFrame(data=knn_probs)

    # Create stacked dataset of both ensemble base model predictions
    S_train = pd.concat([xgb_df, knn_df], axis=1)

    # Define, train, and pickle the adaptive boosting decision stump meta classifier model
    ada = AdaBoostClassifier(algorithm='SAMME.R',
                             learning_rate=1.0, n_estimators=800)

    ada = ada.fit(S_train, y_train)

    joblib.dump(ada, ada_path)

    return


def delete_model():
    """
    The delete model method to clear the existing classifier model.

    :return: void.
    """
    import shutil

    import constants

    dirname = os.path.dirname(__file__)

    path = os.path.join(dirname, f"{constants.MODEL_DIRECTORY}")

    pathExists = os.path.exists(path)

    if pathExists:
        shutil.rmtree(path)
        os.makedirs(path)

    return
