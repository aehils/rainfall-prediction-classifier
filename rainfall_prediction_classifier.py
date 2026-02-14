#!/usr/bin/env python3

# this Machine Learning model will predict whether or not
# it will rain tomorrow based on historical weather data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

def main():
    # load data from Australian Government Bureu of Metrology
    # contains observations of weather metrics for each day from 2008 to 2017
    url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"

    dataset = pd.read_csv(url)
    df = dataset.dropna()
    # print(f"{df.head()} \n {df.info()}")
    df = df.rename(columns={
        'RainToday' : 'RainYesterday',
        'RainTomorrow' : 'RainToday'
    })      # renaming columns in the data to fit a new logical perspective - avoid data leakage

    # focus attention on the region around Melborne
    df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia'])]

    # this is temporal data over full year spans, we need to consider the seasons
    # in fact, the season is likely to be more significant than date
    df['Date'] = pd.to_datetime(df['Date'])     # convert df['Date'] to datetime format
    df['Season'] = df['Date'].apply(date_to_season)     # perform transfomation using date_to_season() helper
    df = df.drop(columns='Date')
    print(df.info(), "\n")

    # separate target from features
    X = df.drop(columns='RainToday', axis=1)
    y = df['RainToday']     #  print(y.value_counts())

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # numerical/categorical feature transformers
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])      # standardise numerical data
    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])      # encode categorical data

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features)
    ])      # combines numerical/categorical transformers into one transformation

    primary_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators' : [50, 100],
        'classifier__max_depth' : [None, 10, 20],
        'classifier__min_samples_split' : [2, 5]
    }       # parameter grid for RandomForest classifier
    param_grid_lr = {
        'classifier__solver' : ['liblinear'],
        'classifier__l1_ratio': [1, 0],
        'classifier__class_weight' : [None, 'balanced']
    }       # parameter grid from LogisticRegression model

    cross_val = StratifiedKFold(n_splits=5, shuffle=True)      # define cross validation method
    model = GridSearchCV(estimator=primary_pipeline,
                         param_grid=param_grid,
                         scoring='accuracy',
                         cv=cross_val, verbose=2)       # grid search on param grid
    model.fit(X_train, y_train)     # fit (read train) model pipeline on training data
    print("\nBest parameters found: ", model.best_params_)
    print("Best cross-validation score: {:.2f}".format(model.best_score_))   

    test_score = model.score(X_test, y_test)
    print(f"Test set score: {test_score:.2%}")
    print("\nClassification Report:")

    # get model pipeline predictions on unseen data
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # plot a confusion matrix for RF classifier
    # confu_matrix = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confu_matrix)
    # disp.plot(cmap='Blues')
    # plt.title('RF Classifier Confusion Matrix')
    # plt.show()

    # get feature importances for the classifier
    cat_feature_names = list(model.best_estimator_.named_steps['preprocessing']
                             .named_transformers_['cat'].named_steps['encoder']
                             .get_feature_names_out(categorical_features))  # first get the encoded categorical feature names
    all_feature_names = numerical_features + cat_feature_names
    feature_importances = model.best_estimator_.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature' : all_feature_names,
        'Importance' : feature_importances
    }).sort_values(by='Importance', ascending=False)

    N = 20  # number of features to display in plot
    top_features = importance_df.head(N)
    # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
    # plt.gca().invert_yaxis()  # invert y-axis to show the most important feature on top
    # plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
    # plt.xlabel('Importance Score')
    # plt.show()

    # REPLACING RANDOMFOREST WITH LINEARREGRESSION
    model.estimator = primary_pipeline  # update estimator with new algo
    model.param_grid = param_grid_lr    # update with new param grid
    primary_pipeline.set_params(classifier=LogisticRegression(random_state=42))


    model.fit(X_train, y_train)     # fit new model to training data
    y_hat = model.predict(X_test)   # make predictions
    
    # COMPARE with previous model
    print(classification_report(y_test, y_hat))
    test_score = model.score(X_test, y_test)
    print(f"\nTest set accuracy: {test_score:.2%}")
    # Generate the confusion matrix 
    conf_matrix = confusion_matrix(y_test, y_hat)
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    # Set the title and labels
    plt.title('Titanic Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # Show the plot
    plt.tight_layout()
    plt.show()      

if __name__ == '__main__':
    main()
