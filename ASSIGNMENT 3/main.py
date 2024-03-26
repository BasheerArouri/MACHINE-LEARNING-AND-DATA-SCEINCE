import numpy as np
import pandas as pd
from scipy.stats import skew, chi2_contingency  # Updated the import statement
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import AdaBoostClassifier


# Function to plot visualizations
def plot_visualizations(df):
    # Create a histogram
    plt.hist(df['ST_Slope'], bins=20, color='skyblue', edgecolor='black')

    plt.title('Histogram of ST_Slope')
    plt.xlabel('ST_Slope')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Pie Chart between ChestPainType and HeartDisease
    plt.figure(figsize=(8, 8))
    df['ChestPainType'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Pie Chart for ChestPainType and HeartDisease')
    plt.ylabel('')  # To remove the default ylabel
    plt.show()

    # Box Plot between HeartDisease and Cholesterol
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='HeartDisease', y='Cholesterol', data=df)
    plt.title('Box Plot between HeartDisease and Cholesterol')
    plt.show()

    # Stacked Bar Chart between HeartDisease and FastingBS
    # Set the figure size
    plt.figure(figsize=(8, 6))

    # Create a stacked bar chart using seaborn
    sns.barplot(data=df, x='HeartDisease', hue='FastingBS', palette='pastel', estimator=lambda x: len(x))

    # Add title and labels
    plt.title('Stacked Bar Chart between HeartDisease and FastingBS')
    plt.xlabel('HeartDisease')
    plt.ylabel('Count')

    # Add legend
    plt.legend(title='FastingBS', loc='upper right')
    plt.show()

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Create KDE plot with fill
    sns.kdeplot(data=df[df['HeartDisease'] == 0]['MaxHR'], label='No Heart Disease', fill=True)
    sns.kdeplot(data=df[df['HeartDisease'] == 1]['MaxHR'], label='With Heart Disease', fill=True)

    # Add title and labels
    plt.title('KDE Curve for MaxHR by HeartDisease')
    plt.xlabel('MaxHR')
    plt.ylabel('Density')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

    # Create a cross-tabulation of HeartDisease and ST_Slope
    cross_tab = pd.crosstab(df['HeartDisease'], df['Age'])

    # Set the figure size
    plt.figure(figsize=(15, 6))

    # Create a heatmap
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu', cbar=True)

    # Add title and labels
    plt.title('Heatmap: HeartDisease vs. Age')
    plt.xlabel('Age')
    plt.ylabel('HeartDisease')

    # Show the plot
    plt.show()

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Create a boxplot
    sns.boxplot(data=df, x='HeartDisease', y='Oldpeak', hue='HeartDisease')

    # Add title and labels
    plt.title('Boxplot: Oldpeak by HeartDisease')
    plt.xlabel('HeartDisease')
    plt.ylabel('Oldpeak')

    # Show the plot
    plt.show()


# Read the dataset and convert it to a dataframe
def read_dataset():
    df = pd.read_csv('heart.csv')
    return df


# Iterate for all the dataset, and check if there is a missing value in each attribute
def print_number_of_missing_values(attribute, missing_values):
    print("For the attribute {}, Number of missing values = {}".format(attribute, missing_values))


def check_null_values(df):
    for attribute in df.columns:
        missing_values = df[attribute].isna().sum()
        print_number_of_missing_values(attribute, missing_values)


def normalize_the_data(df):
    attributes = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    for col in attributes:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def print_statistical_summary(df):
    descriptive_dataframe = dataframe.describe().T
    descriptive_dataframe.drop(columns=['25%', '50%', '75%'], inplace=True)
    descriptive_dataframe.drop(index=['HeartDisease', 'FastingBS'], inplace=True)

    # Continuous attributes
    continuous_attributes = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    continuous_summary = df[continuous_attributes].skew()
    descriptive_dataframe['Skewness'] = continuous_summary
    descriptive_dataframe['Mode'] = df[continuous_attributes].mode().iloc[0]
    descriptive_dataframe['Correlation'] = [df['HeartDisease'].corr(df[attr]) for attr in continuous_attributes]
    print(descriptive_dataframe.to_string())
    print("-" * 100)

    # Nominal attributes
    compute_correlations(df)


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min((k - 1), (r - 1)))


def chi_square_test(df, attribute1, attribute2='HeartDisease'):
    v = cramers_v(df[attribute1], df[attribute2])
    return v


def compute_correlations(df):
    print("Correlation between the nominal features and the output 'HeartDisease' ----->")
    nominal_attributes = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for nominal_attribute in nominal_attributes:
        v_value = chi_square_test(df, nominal_attribute)
        mode_value = df[nominal_attribute].mode().iloc[0]  # Compute mode for the attribute in the DataFrame
        print(f"Correlation between {nominal_attribute} and HeartDisease: {v_value},"
              f" Mode for this feature = {mode_value}")


# function to evaluate all needed performance metrics
def evaluate(labels, predictions):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for p in range(len(predictions)):
        if labels[p] == 1 and predictions[p] == labels[p]:  # if the predicted value is 1 and the same as label (TP)
            true_positive += 1
        elif labels[p] == 1 and predictions[p] == 0:  # if the predicted value is 0 and not the same as label (FP)
            false_negative += 1
        elif labels[p] == 0 and predictions[p] == labels[p]:  # if the predicted value is 0 and the same as label (TN)
            true_negative += 1
        else:  # if the predicted value is 0 and not the same as label (FN)
            false_positive += 1
    accuracy = precision = recall = f1 = 0
    print(f"True Positive = {true_positive}, False Positive = {false_positive}\nTrue Negative = {true_negative}, "
          f"False Negative = {false_negative}")
    accuracy = round(
        (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative), 3)

    precision = round(true_positive / (true_positive + false_positive), 3)

    recall = round(true_positive / (true_positive + false_negative), 3)

    f1 = round((2 * precision * recall) / (precision + recall), 3)

    specificity = round(true_negative / (true_negative + false_positive), 3)
    print(f"Recall = {round(recall * 100, 1)}%, Precision = {round(precision * 100, 1)}%, Accuracy = {round(accuracy * 100, 1)}%, "
          f"Specificity = {round(specificity * 100, 1)}%, F1 = {round(f1 * 100, 1)}%")


def apply_SVM_model(x_train, x_test, y_train, y_test):
    degree_values = [i for i in range(16)]
    # Store mean cross-validation recalls for each degree
    recalls_degree = []

    # Use StratifiedKFold for cross-validation
    cv_degree = StratifiedKFold(n_splits=5)

    for degree_value in degree_values:
        # Create and train the SVM classifier with polynomial kernel and specified degree
        SVM_classifier = svm.SVC(kernel='poly', degree=degree_value, C=0.1)

        # Perform cross-validation on the training set
        # Note: Use recall as the scoring metric
        recalls = cross_val_score(SVM_classifier, x_train, y_train, cv=cv_degree, scoring='recall_macro')
        mean_recall = np.mean(recalls)
        recalls_degree.append(mean_recall)

    # Find the best degree value
    best_degree = degree_values[np.argmax(recalls_degree)]

    # Train the final SVM model on the entire training set using the best parameters
    final_model = svm.SVC(kernel='poly', degree=best_degree, C=0.1)
    final_model.fit(x_train, y_train)

    y_train_pred = final_model.predict(x_train)
    # Calculate metrics
    print("The Metrics on Training set using SVM:")
    evaluate(y_train.to_numpy(), y_train_pred)
    print("--------------------------------------------------------------------")

    # Evaluate the model on the test set
    y_test_pred = final_model.predict(x_test)

    # Calculate metrics
    print("The Metrics on Testing set using SVM:")
    evaluate(y_test.to_numpy(), y_test_pred)

    # Plot cross-validation recalls versus degree values
    plt.plot(degree_values, recalls_degree, marker='o')
    plt.xlabel('Degree')
    plt.ylabel('Cross-Validation Mean Recall')
    plt.title('Cross-Validation Mean Recall for SVM (Poly Kernel, C=0.1, Best Degree={})'.format(best_degree))
    plt.show()
    # Print results
    print(f'Best Degree: {best_degree}')


# Read the CSV file and convert it to a dataframe
dataframe = read_dataset()
df = dataframe.copy() # have copy from data to use it when analyze best model to recover original feature values

print("*************************************| <Data Set"
      "> |***********************************************************")
print(dataframe)


print("*************************************| <Label Count"
      "> |***********************************************************")
print("The Label (Heart Disease) counts: ")
print(dataframe['HeartDisease'].value_counts())


# Check the null values for all the attributes
print("*************************************| <Check Null Values"
      "> |***********************************************************")
check_null_values(dataframe)
print("*************************************| <Some Statistics for "
      "data> |***********************************************************")
print(dataframe.describe().T)
# Print mean, mode, std, variance, and skewness
print_statistical_summary(dataframe)

# Compute and print the results of the correlations between the output feature and other features
compute_correlations(dataframe)

# Plot the exploratory analysis
plot_visualizations(dataframe)

# Normalize the data to converge faster to the goals by Z-score normalization
normalize_the_data(dataframe)

# one hot encoding conversion of categorical features

# find the unique value of each categorical feature to see what we need for encoding technique
# also count values for each categorical feature
print("********************************************| <One Hot Encoding"
      "> |********************************************************")
print("The Sex feature unique values with its number:")
print(dataframe['Sex'].unique())
print(dataframe['Sex'].value_counts())

print("--------------------------------------------------------------------")
print("The ChestPainType feature unique values with its number:")
print(dataframe['ChestPainType'].unique())
print(dataframe['ChestPainType'].value_counts())

print("--------------------------------------------------------------------")
print("The RestingECG feature unique values with its number:")
print(dataframe['RestingECG'].unique())
print(dataframe['RestingECG'].value_counts())

print("--------------------------------------------------------------------")
print("The ExerciseAngina feature unique values with its number:")
print(dataframe['ExerciseAngina'].unique())
print(dataframe['ExerciseAngina'].value_counts())

print("--------------------------------------------------------------------")
print("The ST_Slope feature unique values with its number:")
print(dataframe['ST_Slope'].unique())
print(dataframe['ST_Slope'].value_counts())

# save the label before encoding operation because after encoding we lose the label position in the last column
labels = dataframe['HeartDisease']
dataframeNoLabel = dataframe.drop(columns='HeartDisease')

one_hot_encoded_data = pd.get_dummies(dataframeNoLabel,
                                      columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
                                      dtype=int)
# print("The Date after one hot encoding :- ")
# print(one_hot_encoded_data)
print("********************************************| <Train Test Split"
      "> |********************************************************")
# after this all our data have numeric values, so we can implement any model as we want

# all columns are feature (there is no label because i drop it before) so the features is the hole encoded data frame
features = one_hot_encoded_data
# print(features)

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(f"The train set length = {len(x_train)}\tAnd The test set length = {len(x_test)}")
print("--------------------------------------------------------------------")
print("The training set : \n", x_train)
print("The testing set : \n", x_test)
print("The training label : \n", y_train)
print("The testing label : \n", y_test)

print("********************************************| <1-Nearest Neighbor"
      "> |********************************************************")
# 1-nearest neighbor
one_NN = KNeighborsClassifier(n_neighbors=1, weights='distance')
one_NN.fit(x_train, y_train)  # fitting the model and use it for the predicting the test set examples
predictions = one_NN.predict(x_test)
print("The Metrics for 1-NN :")
evaluate(y_test.to_numpy(), predictions)


# 3-nearest neighbor
print("********************************************| <3-Nearest Neighbor"
      "> |********************************************************")
three_NN = KNeighborsClassifier(n_neighbors=3, weights='distance')
three_NN.fit(x_train, y_train)  # fitting the model and use it for the predicting the test set examples
predictions = three_NN.predict(x_test)
print("The Metrics for 3-NN :")
evaluate(y_test.to_numpy(), predictions)


# AdaBoost Model
print("********************************************| <AdaBoost"
      "> |********************************************************")
estimator = [i for i in range(1, 500, 50)]
# Store mean cross-validation recalls for each degree
recalls_estimator = []

# Use StratifiedKFold for cross-validation
cv_degree = StratifiedKFold(n_splits=5)
for e in estimator:
    Ada_boost = AdaBoostClassifier(n_estimators=e, random_state=42, algorithm='SAMME')
    recalls = cross_val_score(Ada_boost, x_train, y_train, cv=cv_degree, scoring='recall_macro')
    mean_recall = np.mean(recalls)
    recalls_estimator.append(mean_recall)

# Find the best degree value
best_estimator = estimator[np.argmax(recalls_estimator)]
Ada_boost = AdaBoostClassifier(n_estimators=best_estimator, random_state=42, algorithm='SAMME')
Ada_boost.fit(x_train, y_train)

predicted_train = Ada_boost.predict(x_train)
print("The Metrics on Training set using AdaBoost :")
evaluate(y_train.to_numpy(), predicted_train)
print("--------------------------------------------------------------------")

predicted_test = Ada_boost.predict(x_test)
print("The Metrics on Testing set using AdaBoost :")
evaluate(y_test.to_numpy(), predicted_test)
print(f'Best Estimator number for AdaBoost model is : {best_estimator}')

# Plot cross-validation recalls versus number of estimators
plt.plot(estimator, recalls_estimator, marker='x')
plt.xlabel('Estimator Number')
plt.ylabel('Cross-Validation Mean Recall')
plt.title(
    'Cross-Validation Mean Recall for AdaBoost')
plt.show()


# --------------SVM Model------------
print("********************************************| <Support Vector Machine"
      "> |********************************************************")
apply_SVM_model(x_train, x_test, y_train, y_test)


# MLP Model
print("********************************************| <Multi-Layer Perceptron"
      "> |********************************************************")
learning_rate = [0.001, 0.01, 0.1, 1, 10]
# Store mean cross-validation recalls for each degree
recalls_MLP = []

# Use StratifiedKFold for cross-validation
cv_degree = StratifiedKFold(n_splits=5)
for lr in learning_rate:
    MLP = MLPClassifier(hidden_layer_sizes=(5, 4), random_state=42, max_iter=1000, learning_rate_init=lr)
    recalls = cross_val_score(MLP, x_train, y_train, cv=cv_degree, scoring='recall_macro')
    mean_recall = np.mean(recalls)
    recalls_MLP.append(mean_recall)

# Find the best degree value
best_learning_rate = learning_rate[np.argmax(recalls_MLP)]
MLP = MLPClassifier(hidden_layer_sizes=(5, 4), random_state=42, max_iter=1000, learning_rate_init=best_learning_rate)

MLP.fit(x_train, y_train)

predicted_train = MLP.predict(x_train)
print("The Metrics on Training set using MLP :")
evaluate(y_train.to_numpy(), predicted_train)
print("--------------------------------------------------------------------")

predicted_test = MLP.predict(x_test)
print("The Metrics on Testing set using MLP :")
evaluate(y_test.to_numpy(), predicted_test)

# Plot cross-validation recalls versus degree values
plt.plot(learning_rate, recalls_MLP, marker='+')

plt.plot(learning_rate, recalls_MLP, c='r', label='MSE')
plt.xscale('log')  # Set logarithmic scale for the x-axis
plt.xlabel('learning rate')
plt.ylabel('Cross-Validation Mean Recall')
plt.title(
    'Cross-Validation MLP(Best learning rate={})'.format(best_learning_rate))
plt.show()

# Print results
print(f'Best learning rate: {best_learning_rate}')
print("--------------------------------------------------------------------")

index_misclass = []  # list of index of examples that misclassified by MLP
for i in range(len(y_test.to_numpy())):
    if y_test.to_numpy()[i] == 1 and predicted_test[i] == 0:
        index_misclass.append(y_test.index[i])
print("The index of misclassified examples from test set are : ")
print(index_misclass)
for i in range(len(index_misclass)):
    print(df.loc[index_misclass[i]])
    print(".....................................")
