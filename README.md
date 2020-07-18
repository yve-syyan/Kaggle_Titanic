# Kaggle_Titanic

This notebook tackles the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/notebooks) from Kaggle data challenge, whose goal is to predict if a passenger survived based on a series of features. As to July/2020, I scored 0.78 accuracy using RandomForest and reaches top 10% on Kaggle leaderboard.

## Dependencies:
* [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/) for data importing and multi-dimensional computing
* [Scikit-Learn](https://scikit-learn.org/stable/) for machine learning model training and tuning
* [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization

## This Jupyter notebook includes following key features:
* Data preperation
  * Data importation from local files
  * Data imputation
  * Feature extraction
* Data visualization
* Predictive Analysis
  * Trained suprvised models and perfumed hypertuning using GridSearch with cross validation:
    * Logistic Regression
    * Support Vector Machine
    * K-Nearest-Neighbors
    * Naive Bayes
    * Linear Perceptron
    * Stochastic Gradient Descent
    * Decision Tree
    * Random Forest
    
## References:
* [Kaggle-titanic](https://github.com/agconti/kaggle-titanic)
* [Predicting the Survival of Titanic Passengers](https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8)
