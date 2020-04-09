'''
ML Parsing

Initial parsing of the grade submissions. It is meant only for research purposes and not for production environments.

What it does?

    Parses the intial csv files (form csv + grades uploaded in csv format)
    Shows basic info of the data
    Connects form submissions with their associated grades and cleans entry with no grades
    Parses the submission for the CSD Department
    Drops invalid values and fixes any other issues with the dataset
    Produces a final dataset with the form submissions and the grades. Each course found in the dataset is represented as a column. Only the valid courses are kept as found by the OPEN API of AUTH. The dataset contains 89 courses.

*Some columns refer to the same course but they are with different codes (old/new study program)

    This is fixed, old courses are assigned to new course codes.

ATTENTION!!: Some outputs have been deleted for privacy reasons. PLEASE DO NOT UPLOAD OUTPUTS WITH PERSONAL INFO
'''

# Import the modules
# %matplotlib inline
import os
import pandas as pd
import numpy as np
import json

from pandas.io.json import json_normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer

import warnings  # needed for this type of classifier

warnings.simplefilter(action='ignore', category=Warning)


# Global Variables
PATH = "PATH"  # <---------- Change this to path of your project
# Define files
form_csv = "Form_Responses_2020_03_21.csv"
folder_with_grade_csvs = "GradeParser/csv/"
# Json with the valid courses as got from OPEN AUTH API (https://ws-ext.it.auth.gr/)
valid_courses_json = "courses_ids_600000014.json"


def main():
    # Reading the form submissions (download csv file)
    submissions_csv = pd.read_csv(form_csv)

    # Create an extract column to store the paths of the csvs with the grades
    csv = [None] * (submissions_csv.shape[0])
    submissions_csv["CSV"] = pd.Series(csv).to_numpy()

    find_associated_csv(submissions_csv)

    # Nan CSV entries
    # Drop those entries with no csv associated
    submissions_csv.dropna(subset=['CSV'], inplace=True)

    department = "Πληροφορική ΑΠΘ"
    data = csd_parsing(submissions_csv, department)

    ects = pd.read_csv(PATH + "subjects_ects.csv")
    ects_dict = ects.set_index('Subjects')['Ects'].to_dict()

    ml_features(data, ects_dict)

    rename_columns(data)

    data = handle_categorical(data)

    # Write to file
    data.to_csv(PATH + "csd_2020_renamed.csv", index=False)


# Find associated csv - Function
def find_associated_csv(submissions_csv):
    for r, d, f in os.walk(folder_with_grade_csvs):
        for file in f:
            if ".csv" in file:  # For each csv
                without_ext = file.replace("_results.csv", "")  # Remove extensions added by the parser\n",
                # Set the csv found \n",
                submissions_csv.loc[
                    submissions_csv["Filename"].str.contains(without_ext, regex=False), "CSV"] = r + file


# Parsing for Departments - Function
def csd_parsing(submissions_csv, department):
    # CSD Department - Grade Statistics
    # Get the CSD department
    csd = submissions_csv[submissions_csv['Σχολή Φοίτησης'] == department]

    # Concat all grades (all the grade csvs together)
    csd_csv_files = csd['CSV'].to_numpy()
    li = []

    # pd.read_csv(csd_csv_files[5],encoding='utf-8')
    colnames = ["C_ID", "C_NAME", "TYPE", "EXAM_YEAR", "EXAM_PERIOD", "ECTS", "DM", "FACTOR", "GRADE"]
    # coltypes = {"C_ID":str,"C_NAME":str,"TYPE":str,"EXAM_YEAR":str,"EXAM_PERIOD":str,"ECTS":np.int32,"DM":np.int32,"FACTOR":np.int32,"GRADE":np.float64}
    for csv_file in csd_csv_files:
        f = open(csv_file, encoding='utf-8')
        df = pd.read_csv(f, index_col=None, header=None, names=colnames)
        li.append(df)

    csd_grades = pd.concat(li, axis=0, ignore_index=True)

    # Find more information on the courses
    unique_courses = csd_grades["C_ID"].unique()
    csd_grades['C_ID'] = csd_grades['C_ID'].str.replace(' ', '', regex=True)
    csd_grades["C_ID"].value_counts()

    # Checking more info
    unique_names = csd_grades.groupby('C_NAME')['C_ID'].value_counts()

    csd_grades = fix_issues(csd_grades)
    unique_names = csd_grades.groupby('C_NAME')['C_ID'].value_counts()
    #print_full(unique_names)

    colnames = ["C_ID", "C_NAME", "TYPE", "EXAM_YEAR", "EXAM_PERIOD", "ECTS", "DM", "FACTOR", "GRADE"]
    unique_courses = csd_grades["C_ID"].unique()
    course_dictionary = dict()
    # Create a dictionary for each course, each row represents a student
    for course in get_valid_courses():
        course_dictionary[course] = [-1] * len(csd_csv_files)
    for i in range(0, len(csd_csv_files)):
        csv_file = csd_csv_files[i]
        f = open(csv_file, encoding='utf-8')
        df = pd.read_csv(f, index_col=None, header=None, names=colnames)
        df = fix_issues(df)
        df_courses = df["C_ID"].to_numpy()
        for course_id in df_courses:
            if course_id in course_dictionary:
                course_dictionary[course_id][i] = df[df["C_ID"] == course_id]["GRADE"].to_numpy()[0]
            #else:
                #print("Not valid course:" + course_id + '-' + csd_grades[csd_grades["C_ID"] == course_id]
                #['C_NAME'].values)
    csd_courses_df = pd.DataFrame.from_dict(course_dictionary)

    # Fill nan values with -1
    csd_courses_df.fillna(-1, inplace=True)
    # Replace blank values with -1
    csd_courses_df.replace(r'^\s*$', -1, regex=True, inplace=True)
    # Replace ΕΠΙΤ with 10
    csd_courses_df.replace(' ΕΠΙΤ', 10, inplace=True)

    # Change to type flaot
    csd_courses_df = csd_courses_df.astype(np.float64)

    #### Concatenate CSD Student + Grades ####

    # Remove unnecessary fields for final dataset
    csd_final = csd.drop(['Timestamp', 'Σχολή Φοίτησης',
                          'Υποβολή αναλυτικής βαθμολογίας',
                          'Συναινώ στη χρήση των στοιχείων μου από την ομάδα του Grade++',
                          'Αν θες να λάβεις πρώτος πρόσβαση στην εφαρμογή, συμπλήρωσε το email σου:',
                          'Filename', 'CSV'], axis=1)
    csd_final.reset_index(drop=True, inplace=True)

    # Concat with grades
    ldf = [csd_final, csd_courses_df]
    csd_final = pd.concat(ldf, axis=1)
    return csd_final


# Engineering ML Features (GPA and ECTS) - Function
def ml_features(data, ects_dict):
    data_courses = data.iloc[:, 10:]
    ects_total = 0
    total = 0
    gpa = 0
    number = 0
    gpa_list = []
    ects_list = []
    ects_needed = []
    number_of_prior_courses = []
    for i in range(0, 43):
        for column in data_courses.columns[:]:
            temp = data_courses.at[i, column]
            if temp >= 5:
                number = number + 1
                total = total + temp * ects_dict.get(column)
                ects_total = ects_total + ects_dict.get(column)
        # print('GPA: ', total / ects_total , 'ECTS: ', ects_total)
        number_of_prior_courses.append(number)
        ects_list.append(ects_total)
        ects_needed.append(240 - ects_total)
        gpa_list.append(total / ects_total)
        ects_total = 0
        total = 0
        number = 0


# Rename Columns - Function
def rename_columns(data):
    # Change Courses Names
    with open(valid_courses_json, "r") as json_file:
        json_file = json.load(json_file)
    coded_courses = pd.json_normalize(json_file['courses'])
    del coded_courses['ccoursecode']
    course_dict = coded_courses.set_index('coursecode')['courseId'].to_dict()

    # Change the rest of the columns' names
    data.rename(columns={"Ηλικία": "age", "Φύλο": "gender", "Επέλεξα τη σχολή μου διότι:": "reason",
                         "Κατά μέσο όρο την εβδομάδα, διαβάζω:": "study_time",
                         "Μέσα στο εξάμηνο, παρακαλουθώ:": "lectures",
                         "Υπήρξε ανάγκη για φροντηστηριακή βοήθεια σε κάποιο μάθημα έως τώρα;": "private",
                         "Μετά το πτυχίο, θα ήθελα να ακολουθήσω:": "postgraduate",
                         "Ποιο από τα παρακάτω ισχύει;": "roomates",
                         "Η σχολή απέχει από το σπίτι μου:": "distance",
                         "Ασχολούμαι εβδομαδιαία με:": "hobbies"}, inplace=True)
    for key in course_dict.keys():
        data.rename(columns={key: str(course_dict[key]).strip()}, inplace=True)


# Fixing Issues - Function
def fix_issues(df):
    df['C_ID'] = df['C_ID'].str.replace(' ', '', regex=True)  # Removing blank cases

    df['C_NAME'] = df['C_NAME'].str.replace('*', '', regex=True)  # Removing stars
    df['C_NAME'] = df['C_NAME'].str.strip()  # Removing leading and trailing spaces

    # Fixing issue with parser
    df['C_NAME'] = df['C_NAME'].replace('ΑΝΤΙΚΕΙΜΕΝΟΣΤΡΕΦΗΣ', 'ΑΝΤΙΚΕΙΜΕΝΟΣΤΡΕΦΗΣ ΠΡΟΓΡΑΜΜΑΤΙΣΜΟΣ', regex=False)
    df['C_NAME'] = df['C_NAME'].replace('ΘΕΩΡΙΑ ΓΛΩΣΣΩΝ ΚΑΙ', 'ΘΕΩΡΙΑ ΓΛΩΣΣΩΝ ΚΑΙ ΑΥΤΟΜΑΤΩΝ', regex=False)
    df['C_NAME'] = df['C_NAME'].replace('ΤΕΧΝΙΚΕΣ ΔΟΜΗΣΗΣ', 'ΤΕΧΝΙΚΕΣ ΔΟΜΗΣΗΣ ΔΕΔΟΜΕΝΩΝ', regex=False)

    # Grouping old courses with new ones with the same name - CHANGED TO MANUALLY
    unique_courses = df['C_NAME'].unique()
    for course in unique_courses:
        course_grades = df[df['C_NAME'] == course]
        if course_grades['C_ID'].nunique() > 1:
            course_codes = course_grades['C_ID'].value_counts().index.values
            for i in range(1, len(course_codes)):
                df['C_ID'] = df['C_ID'].replace(course_codes[i], course_codes[0], regex=False)

    ## Manual Work ##
    df['C_NAME'] = df['C_NAME'].replace('ΑΝΑΓΝΩΡΙΣΗ ΠΡΟΤΥΠΩΝ', 'ΑΝΑΓΝΩΡΙΣΗ ΠΡΟΤΥΠΩΝ -ΣΤΑΤΙΣΤΙΚΗ ΜΑΘΗΣΗ', regex=False)
    df['C_ID'] = df['C_ID'].replace('NDM-06-03', 'NDM-06-04', regex=False)
    df['C_NAME'] = df['C_NAME'].replace('ΑΝΩΤΕΡΑ ΜΑΘΗΜΑΤΙΚΑ Ι', 'ΜΑΘΗΜΑΤΙΚΗ ΑΝΑΛΥΣΗ Ι', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-01-01', 'NCO-01-01', regex=False)
    df['C_NAME'] = df['C_NAME'].replace('ΑΝΩΤΕΡΑ ΜΑΘΗΜΑΤΙΚΑ ΙΙ', 'ΜΑΘΗΜΑΤΙΚΗ ΑΝΑΛΥΣΗ ΙΙ', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-02-01', 'NCO-02-01', regex=False)
    df['C_NAME'] = df['C_NAME'].replace('ΕΙΣΑΓΩΓΗ ΣΤΟΥΣ ΥΠΟΛΟΓΙΣΤΕΣ', 'ΕΙΣΑΓΩΓΗ ΣΤΗΝ ΠΛΗΡΟΦΟΡΙΚΗ', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-01-02', 'NCO-01-02', regex=False)
    df['C_NAME'] = df['C_NAME'].replace('ΘΕΩΡΙΑ ΓΛΩΣΣΩΝ ΚΑΙ ΑΥΤΟΜΑΤΩΝ', 'ΘΕΩΡΙΑ ΥΠΟΛΟΓΙΣΜΟΥ', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-02-04', 'NCO-02-05', regex=False)
    df['C_NAME'] = df['C_NAME'].replace('ΝΕΥΡΩΝΙΚΑ ΔΙΚΤΥΑ', 'ΝΕΥΡΩΝΙΚΑ ΔΙΚΤΥΑ - ΒΑΘΙΑ ΜΑΘΗΣΗ', regex=False)
    df['C_ID'] = df['C_ID'].replace('NDM-07-03', 'NDM-07-05', regex=False)
    df['C_NAME'] = df['C_NAME'].replace('ΠΙΘΑΝΟΤΗΤΕΣ KAI ΣΤΑΤΙΣΤΙΚΗ', 'ΠΙΘΑΝΟΤΗΤΕΣ & ΣΤΑΤΙΣΤΙΚΗ', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-02-02', 'NCO-02-02', regex=False)
    df['C_NAME'] = df['C_NAME'].replace('ΣΧΕΔΙΑΣΗ ΑΛΓΟΡΙΘΜΩΝ', 'ΑΛΓΟΡΙΘΜΟΙ', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-04-05', 'NCO-04-03', regex=False)
    df['C_NAME'] = df['C_NAME'].replace('ΣΧΕΔΙΑΣΗ ΓΛΩΣΣΩΝ ΚΑΙ ΜΕΤΑΓΛΩΤΤΙΣΤΕΣ', 'ΣΧΕΔΙΑΣΗ ΓΛΩΣΣΩΝ ΠΡΟΓΡΑΜΜΑΤΙΣΜΟΥ ',
                                        regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-04-03', 'NCO-04-01', regex=False)

    # More manual
    df['C_ID'] = df['C_ID'].replace('CO-01-05', 'NCO-01-05', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-01-06', 'NCO-01-04', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-01-03', 'NCO-01-03', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-02-05', 'NCO-02-04', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-04-02', 'NCO-04-05', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-03-03', 'NCO-03-04', regex=False)
    df['C_ID'] = df['C_ID'].replace('CO-04-04', 'NCO-04-02', regex=False)

    return df


# Printing the full dataframe - Function
def print_full(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):  # more options can be specified also
        print(df)


# Get's the real courses offered as got from the AUTH Open Data API (https://ws-ext.it.auth.gr/) - Function
def get_valid_courses():
    with open(valid_courses_json) as json_file:
        data = json.load(json_file)
        df = pd.DataFrame(data['courses'])
    json_file.close()
    return df['coursecode'].values


# Categorical Values Handling - Function
def handle_categorical(data):
    # Let's split to characteristics and courses
    data_characteristics = data.iloc[:, :10]
    data_courses = data.iloc[:, 10:]

    # Categorical values that maintain the scaling properties "study_time","lectures","postgraduate","distance"
    # Study_time
    data_characteristics["study_time"] = data_characteristics["study_time"].replace(
        {"0 - 2 ώρες": 0.0, "2 - 5 ώρες": 1.0, "> 5 ώρες": 2.0})
    data_characteristics["study_time"] = pd.to_numeric(data_characteristics["study_time"])

    # Lectures
    data_characteristics["lectures"] = data_characteristics["lectures"].replace(
        {"Λιγότερες από τις μισές διαλέξεις": 0.0,
            "Περίπου τις μισές διαλέξεις": 1.0,
            "Παραπάνω από τις μισές διαλέξεις": 2.0,
            "Όλες τις διαλέξεις": 3.0})
    data_characteristics["lectures"] = pd.to_numeric(data_characteristics["lectures"])

    # Postgraduate
    data_characteristics["postgraduate"] = data_characteristics["postgraduate"].replace({"Τίποτα από τα δύο": 0.0,
                                                                                         "Μεταπτυχιακές Σπούδες": 1.0,
                                                                                         "Διδακτορικές Σπουδές": 2.0})
    data_characteristics["postgraduate"] = pd.to_numeric(data_characteristics["postgraduate"])

    # Distance
    data_characteristics["distance"] = data_characteristics["distance"].replace({"< 10 λεπτά": 0.0,
                                                                                 "10 - 25 λεπτά": 1.0,
                                                                                 "25 - 45 λεπτά": 2.0,
                                                                                 "> 45 λεπτά": 3.0})
    data_characteristics["distance"] = pd.to_numeric(data_characteristics["distance"])

    # Gender
    data_characteristics["gender"] = data_characteristics["gender"].replace({"Κορίτσι": 1, "Αγόρι": 0})
    data_characteristics["gender"] = pd.to_numeric(data_characteristics["gender"])

    # private
    data_characteristics["private"] = data_characteristics["private"].replace({"Ναι": 0,
                                                                               "Όχι": 1})
    data_characteristics["private"] = pd.to_numeric(data_characteristics["private"])

    # One-hot encoder columns (only roomates)

    ohe_columns = ["roomates"]

    full_pipeline = ColumnTransformer([
        ('one_hot', OneHotEncoder(), ohe_columns)
    ])

    roomates = full_pipeline.fit_transform(data_characteristics)
    # Concat with  data_characteristics with roomates
    roomates_df = pd.DataFrame(roomates.toarray(), columns=['family', 'alone', 'friend', 'siblings'], dtype=np.int8)
    data_characteristics_updated = pd.concat([data_characteristics.drop("roomates", axis=1), roomates_df], axis=1)

    # Convert string cell with multiple values to list
    acceptable_hobbies = ["Σειρές / Ταινίες", "Αθλητισμό", "Video Games", "Ξένη γλώσσα", "Εθελοντισμός"]
    for student in range(0, data_characteristics.shape[0]):
        data_characteristics['reason'][student] = data_characteristics['reason'][student].split(
            ", ")  # There is a space after each comma
        # Hobbies transformation

        hobbies_list = data_characteristics['hobbies'][student].split(", ")
        for i in range(0, len(hobbies_list)):
            if hobbies_list[i] not in acceptable_hobbies:
                hobbies_list[i] = "Άλλο"
        data_characteristics['hobbies'][student] = hobbies_list

        # Multilabel Binarizer

        mlb = MultiLabelBinarizer()
        hobbies = mlb.fit_transform(data_characteristics['hobbies'])
        #         print(mlb.classes_)
        reasons = mlb.fit_transform(data_characteristics['reason'])
        #         print(mlb.classes_)

    # Convert to data_characteristics frame and concat
    hobbies_df = pd.DataFrame(hobbies, columns=["vgames", "other", "sports", "volunteer", "languange", "movies"])
    data_characteristics_updated = pd.concat([data_characteristics_updated.drop(["hobbies"], axis=1), hobbies_df],
                                             axis=1)

    reasons_df = pd.DataFrame(reasons, columns=["quality", "choice", "subject", "parents", "career"])
    data_characteristics_updated = pd.concat([data_characteristics_updated.drop(["reason"], axis=1), reasons_df],
                                             axis=1)

    full_data = pd.concat([data_characteristics_updated, data_courses], axis=1)

    return full_data


main()
