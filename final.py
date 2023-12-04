import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
data = pd.read_csv('train.csv')
train = data
train_xs = train.drop(columns = "Transported")
train_ys = train['Transported']
test_xs = pd.read_csv('test.csv')
train_xs.dtypes
print(train_xs)

class OneHotEncodeCategorical(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #print(temp)
        columns_to_drop = ["Name","PassengerId"]
        temp = X.drop(columns = columns_to_drop)
        temp.fillna(value=0,inplace=True)

        X_copy = temp.copy()
        cabin_info = temp['Cabin'].str.extract(r'(?P<Deck>[A-Za-z])/(?P<Number>\d+)/(?P<Side>[PS])')
        #adding deck and number give a nan warning for test scores
        #X_copy = pd.concat([X_copy, cabin_info['Deck']], axis=1)
        #X_copy = pd.concat([X_copy, cabin_info['Number']], axis=1)

        X_copy = pd.concat([X_copy, cabin_info['Side']], axis=1)
        X_copy = X_copy.drop(columns="Cabin")  
        temp = X_copy;

        categorical_columns = temp.select_dtypes(include=['object']).columns
        X_encoded = pd.get_dummies(temp, columns=categorical_columns)
        print(X_encoded.columns.tolist())

        return X_encoded


gradientboosting_pipeline = Pipeline([
    ('ordinal_encoder', OneHotEncodeCategorical()),
    ('scaler',MinMaxScaler()),
    ('gradient_boosting', GradientBoostingClassifier())
])

gradientboosting_grid = {
    'gradient_boosting__subsample': [0.5,0.6,0.7,0.8,0.9,1], #The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting
    'gradient_boosting__learning_rate': [0.1,0.3,0.5],  #learning rate shrinks the contribution of each tree
    'gradient_boosting__n_estimators': [50,60,70,80,90,100],  #number of boosting stages, larger number tends to do better
    'gradient_boosting__max_depth': [3,4,5,6,7],  #limits number of nodes in tree
}
gradientboosting_search = GridSearchCV(gradientboosting_pipeline, gradientboosting_grid, scoring='accuracy', n_jobs=-1)
gradientboosting_search.fit(train_xs, train_ys)

gradientboosting_params = gradientboosting_search.best_params_
gradientboosting_score = gradientboosting_search.best_score_
print(f"Accuracy: {gradientboosting_score}")
print(f"Best params: {gradientboosting_params}\n")

best_gradientboosting = gradientboosting_search.best_estimator_

predicted_values = best_gradientboosting.predict(test_xs)
passenger_ids = test_xs['PassengerId'].reset_index(drop=True)

result_df = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': predicted_values})

print(result_df)
result_df.to_csv('predicted_results.csv', index=False)
