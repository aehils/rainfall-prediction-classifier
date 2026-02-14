#!/usr/bin/env python3

# this Machine Learning model will predict whether or not
# it will rain tomorrow based on historical weather data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
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
    y = df['RainToday']

    print(y.value_counts())



    


if __name__ == '__main__':
    main()
