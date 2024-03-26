import pandas as pd

# ------------------------------This is the second part------------------------------
dataframe = pd.read_csv("cars.csv")
for feature in dataframe:
    # Series type for missing values
    missing_values = pd.isnull(dataframe[feature])
    # I want to convert it string type
    number_of_missing_values = missing_values.to_string().count('True')
    print("For Feature {}".format(feature) + ", number of missing values are {}".format(number_of_missing_values))
