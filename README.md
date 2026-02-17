# Rainfall Prediction: Will It Rain Tomorrow?

I built a Random Forest classifier pipeline and optimised using grid search cross-validation.
Classification Report and the GridSearchCV function from `sklearn` provide pretty convenient methods to evaluate the model's performance.

``` bash
Classification Report:
              precision    recall  f1-score   support

          No       0.86      0.95      0.90      1154
         Yes       0.75      0.51      0.61       358

    accuracy                           0.84      1512
   macro avg       0.81      0.73      0.75      1512
weighted avg       0.84      0.84      0.83      1512

```
``` bash
Best parameters found:  {'classifier__max_depth': None, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100}
Best cross-validation score: 0.85
Test set score: 84.39%
```

I went a little further to update the pipeline with a new classifier: a Logistic Regressor. 
Though I run similar evaluation metrics as the ensemble method and compare each model pipeline, it was not about finding the better algorithm.
This work is about understanding how ML pipelines can be optimised by tuning hyperparameters through Grid Search and model validation.

The dataset used in this repo is sourced from Kaggle at 
[https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package?resource=download&select=weatherAUS.csv](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package?resource=download&select=weatherAUS.csv)

## How Can You Use It Yourself?
You can just download the Python script and run it! Provided you have Python installed on your machine, of course.
In the shebang I specify the Python3.0 compiler which you can edit, but I'd expect that anyone remotely curious about running this likely already has this version of py.

Please keep in mind that I constrained the geographic region to the area surrounding Melbourne.
I encourage curious cats out there to pick other parts of Australia to see if maybe the model performances differ (that would be interesting tbh).
