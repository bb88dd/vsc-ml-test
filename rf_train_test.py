from sklearn.datasets import load_iris
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def get_data():
        # Load iris data set.
        iris = load_iris()
        # Extract features to dataframe and add column names.
        features = pd.DataFrame(iris.data)
        features.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        # Extract target to dataframe and add column name.
        target = pd.DataFrame(iris.target)
        target.columns = ['target']
        # Option to concat these together for data exploration.
        df = pd.concat([features, target], axis=1)
        # Option to import from URL: df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

        return features, target


def preprocess_data(features, target):
        # Split into train and test data.
        x_train, x_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=0.2,
                stratify=target,
                random_state=42
        )
        # Scale features.
        numerical_attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        passthrough_attributes = []
        full_pipeline = ColumnTransformer(
                transformers=[
                        ("numerical", StandardScaler(), numerical_attributes)
                ]
        )
        x_train_prepared = full_pipeline.fit_transform(x_train)
        # Scale test data (using previously created pipeline).
        x_test_prepared = full_pipeline.transform(x_test)

        return x_train_prepared, y_train, x_test_prepared, y_test


def train_model(x_train, y_train, cv_folds=5, n_iter=10):
        # Define search grid.
        param_grid = [
                {'bootstrap': [True, False],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'max_features': ['sqrt'],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
        ]
        # Use randomized grid search to find the best performing model according to AUC.
        rf_clf = RandomForestClassifier(random_state=42)
        grid_search = RandomizedSearchCV(
                rf_clf,
                param_distributions=param_grid,
                cv=cv_folds,
                scoring='roc_auc',
                return_train_score=True,
                n_iter=n_iter,
        )
        grid_search.fit(x_train, y_train)
        model = grid_search.best_estimator_

        return model


def test_model(model, x_test, y_test):
    """
    Test model performance on accuracy, precision, recall and AUC.
    :param model: Churn classification model.
    :param validation_prepared: Validation data set ready to be used by the model.
    :param validation_labels: Labels for the validation data set (retained = 1, churned = 0).
    :return: Dataframe of model performance metrics.
    """
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    print('Accuracy :', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)

    return accuracy, precision, recall


# Execute functions to train RF model and return performance metrics.
x, y = get_data()
x_train, y_train, x_test, y_test = preprocess_data(x, y)
model = train_model(x_train, y_train)
test_model(model, x_test, y_test)
