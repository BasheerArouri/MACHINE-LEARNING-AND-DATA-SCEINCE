import pandas as pd
contents = pd.read_csv("cars.csv")
# print(contents.to_string()) -------> print the entire dataframe

#This is just to print the number of features and examples in the dataset
print(contents)
