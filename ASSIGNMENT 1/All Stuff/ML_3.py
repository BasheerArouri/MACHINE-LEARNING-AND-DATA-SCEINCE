import pandas as pd

# ------------------------------This is the third part------------------------------
dataframe = pd.read_csv("cars.csv")
# I calculated the mode for the columns and replaced each missing value with this mode value
modes_dataframe = dataframe.mode()  # returns a dataframe contains mode for each column
for mode_for_feature in modes_dataframe:
    current_mode = modes_dataframe[mode_for_feature][0]  # Getting the mode for each column for this dataframe
    # fill the missing value with mode for this feature
    dataframe[mode_for_feature] = dataframe[mode_for_feature].fillna(current_mode)

# This is just to confirm that the missing data has been completed#
for feature in dataframe:
    # Series type for missing values
    missing_values = pd.isnull(dataframe[feature])
    # I want to convert it string type
    number_of_missing_values = missing_values.to_string().count('True')
    print("For Feature {}".format(feature) + ", number of missing values are {}".format(number_of_missing_values))
