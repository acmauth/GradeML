import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display

from sklearn.model_selection import LeaveOneOut
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import pickle

import warnings  # needed for this type of classifier
warnings.simplefilter(action='ignore', category=Warning)


PATH = "PATH"  # <---------- Change this to path of your project.


def main():
    data = pd.read_csv(PATH + "csd_2020_renamed.csv")
    characteristics_cols, courses_cols = split_data(data)
    course_case = -1
    feature_case = 1
    course_columns = select_courses(course_case, courses_cols, data)
    selected_columns = select_features(feature_case, characteristics_cols, courses_cols)
    
    # Select only those columns
    data_selected = data.loc[:, selected_columns]
    training(course_columns, data_selected, data)

    X, y = training(course_columns, data_selected, data)
    #corellation_table(data_selected)

    # Save the models
    persistence(data_selected, course_columns)


def split_data(data):
    # Let's split to characteristics and courses
    data_characteristics = data.iloc[:, :22]
    data_courses = data.iloc[:, 22:]
    return data_characteristics.columns, data_courses.columns


def select_courses(case, courses_cols, data):
    if case == -1:  # All Courses
        course_columns = courses_cols
    else:  # The Courses that the students passed are > case (given number)
        course_columns = select_some_subjects(data, courses_cols, case)
    return course_columns


def select_features(case, characteristics_cols, course_columns):
    selected_columns = characteristics_cols.to_list() + course_columns.to_list()
    if case == 1:  # Only Courses
        selected_columns = course_columns
    if case == 2:  # Only Characteristics
        selected_columns = characteristics_cols
    if case == 3:  # Characteristics + Courses
        selected_columns = characteristics_cols.to_list() + course_columns.to_list()
    return selected_columns


def select_some_subjects(data, courses_cols, number_of_students_passed):
    # Creating dictionary with courses as keys and number of students that passed the subject as values
    new_dict = {new_list: [] for new_list in range(0)}
    total_sum = 0
    iterations = 0
    for column in courses_cols:
        temp = data[data[column] > -1].shape[0]
        if temp != 0:
            total_sum = total_sum + temp
            iterations = iterations + 1
        new_dict[column] = temp
    mean = total_sum / iterations
    # List with only the courses we are going to use for the model
    alist = []
    for key in new_dict.keys():
        if new_dict[key] > number_of_students_passed:
            alist.append(key)
    return alist


def training(course_columns, data_selected, data):
    sum_error = 0
    sum_squared_error = 0
    for course_selected in course_columns:  # For each course
        print(course_selected)
        errors = []
        sq_errors = []
        if course_selected in data_selected:
            X = data_selected.drop(course_selected, axis=1, inplace=False)
        else:
            X = data_selected
        y = data.loc[:, course_selected]
        loo = LeaveOneOut()
        xgb = XGBRegressor(objective='reg:squarederror')
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #         print(X_train)
            xgb.fit(X_train, y_train)
            predictions = xgb.predict(X_test)
            errors.append(mean_absolute_error(y_test, predictions))
            sq_errors.append(mean_squared_error(y_test, predictions))

        sum_error += np.mean(errors)
        sum_squared_error += np.mean(sq_errors) ** (1 / 2)
        print("MAE:" + str(np.mean(errors)))
        print("RMSE:" + str((np.mean(sq_errors)) ** (1 / 2)))

    print("Mean MAE:" + str(sum_error / (len(course_columns))))
    print("Mean RMSE:" + str(sum_squared_error / (len(course_columns))))
    return X, y


def corellation_table(data_selected):
    corrmat = data_selected.corr(method="pearson")
    top_corr_features = corrmat.index
    plt.figure(figsize=(35, 35))
    g = sns.heatmap(data_selected[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()


def persistence(data_selected, course_columns):
    for prediction_course in course_columns:
        data_persistence_selected = data_selected[data_selected[prediction_course] >= 0]
        X = data_persistence_selected.drop(prediction_course,axis=1,inplace=False)
        y = data_persistence_selected.loc[:,prediction_course]
        xgb_model = XGBRegressor(objective = 'reg:squarederror')
        xgb_model.fit(X,y)
        pickle.dump(xgb_model, open(PATH + "models/version1/" + prediction_course +".dat", "wb"))

    print(xgb_model.get_booster().feature_names)

main()
