import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.read_csv("cars.csv")

# ------------------------------This is the third part------------------------------
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

# ------------------------------This is the forth part------------------------------
# First, make the Europe dataframe that contains just the europe origin with corresponding mpgs
condition = dataframe['origin'] == 'Europe'
europe_dataframe = dataframe[condition]

# Second, make the Asia dataframe that contains just the Asia origin with corresponding mpgs
condition = dataframe['origin'] == 'Asia'
Asia_dataframe = dataframe[condition]

# Third, make the USA dataframe that contains just the USA origin with corresponding mpgs
condition = dataframe['origin'] == 'USA'
USA_dataframe = dataframe[condition]

fig, ax = plt.subplots()
# Plot box plot for each DataFrame on the same graph
ax.boxplot([europe_dataframe['mpg'], Asia_dataframe['mpg'], USA_dataframe['mpg']], labels=['Europe', 'Asia', 'USA'])
plt.ylabel('mpg')
# Setting the title
plt.title('mpg For The Three Countries')
# Show the plot
plt.show()

# ------------------------------This is the fifth part------------------------------

# First, the histogram for the acceleration
plt.hist(dataframe['acceleration'], bins=10, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Frequency')
plt.ylabel('Acceleration')
plt.title('Histogram of Acceleration')
plt.show()

# #Second, the histogram for the horsepower
plt.hist(dataframe['horsepower'], bins=10, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Frequency')
plt.ylabel('Horsepower')
plt.title('Histogram of Horsepower')
plt.show()

# Third, the histogram for the mpg
plt.hist(dataframe['mpg'], bins=10, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Frequency')
plt.ylabel('mpg')
plt.title('Histogram of mpg')
plt.show()

# ------------------------------This is the Sixth part------------------------------

# I found the skewness for acceleration, horsepower and mpg features
skewness_series = dataframe[['acceleration', 'horsepower', 'mpg']].skew()
print("\n\nThe Skewness for acceleration is {}".format(skewness_series['acceleration']))
print("The Skewness for horsepower is {}".format(skewness_series['horsepower']))
print("The Skewness for mpg is {}".format(skewness_series['mpg']))

# ------------------------------This is the Seventh part------------------------------

dataframe.plot(kind='scatter', x='horsepower', y='mpg', title='Scatter Plot Between Horsepower and mpg')
plt.show()
correlation = dataframe['horsepower'].corr(dataframe['mpg'])
print("\n\nCorrelation Coefficient between mpg and horsepower features is {}".format(correlation))

# ------------------------------This is the Eighth part------------------------------
# Create the closed form solution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataframe['X0'] = [1 for _ in range(398)]  # make a feature with ones for the intercept
X = dataframe[['X0', 'horsepower']].values  # make the data matrix from X0 and Horsepower features
Y = dataframe['mpg'].values  # make an array from the target values
XT = np.transpose(X)  # make a transpose matrix
XT_X = np.dot(XT, X)  # make a dot product between X and X transpose
XT_X_inv = np.linalg.inv(XT_X)
XT_X_inv_XT = np.dot(XT_X_inv, XT)
W = np.dot(XT_X_inv_XT, Y)  # make the weights from the closed from solution (slope and the intercept)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create the predicted output
predicted_output = W[0] + W[1] * dataframe['horsepower']

# Plot the scatter plot
dataframe.plot(kind='scatter', x='horsepower', y='mpg', label='Horsepower Data')

# Plot the regression line
plt.plot(dataframe['horsepower'], predicted_output, color='red', label='Regression Line')

# Add labels and a legend
plt.xlabel('Horsepower')
plt.ylabel('mpg')
plt.legend()
plt.show()
print("\n\nW[0] = {}".format(W[0]))
print("W[1] = {}".format(W[1]))

# ------------------------------This is the Ninth part------------------------------
# Create the closed form solution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataframe['X_square'] = dataframe['horsepower'] ** 2  # make X^2 to learn the model
X = dataframe[['X0', 'horsepower', 'X_square']].values  # make the data matrix from X0, X and X^2 features
Y = dataframe['mpg'].values  # make an array from the target values
XT = np.transpose(X)  # make a transpose matrix
XT_X = np.dot(XT, X)  # make a dot product between X and X transpose
XT_X_inv = np.linalg.inv(XT_X)
XT_X_inv_XT = np.dot(XT_X_inv, XT)
W = np.dot(XT_X_inv_XT, Y)  # make the weights from the closed from solution (slope and the intercept)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create the predicted output after sorting
X = dataframe['horsepower'].sort_values()
X_square = X ** 2
predicted_output = W[0] + W[1] * X + W[2] * X_square

# Plot the scatter plot
dataframe.plot(kind='scatter', x='horsepower', y='mpg', label='Horsepower Data')

# Plot the regression line
plt.plot(X, predicted_output, color='red', label='Regression Line')

# Add labels and a legend
plt.xlabel('Horsepower')
plt.ylabel('mpg')
plt.legend()
plt.show()
print("\n\nFor degree = 2 ------> W[0] = {}, W[1] = {}, W[2] = {}".format(W[0], W[1], W[2]))


# ------------------------------This is the Tenth part------------------------------
def get_cost(scaled_values_for_X0, scaled_values_for_horsepower, target_values, n, W):
    cost = 0
    for i in range(n):
        f_x = W[0] * scaled_values_for_X0[i] + W[1] * scaled_values_for_horsepower[i]
        cost = cost + (f_x - target_values[i]) ** 2
    cost = cost / n
    return cost


# Prepare the data
# I made a scaling by Z-score for the values to converge faster to the weights
mean = dataframe['horsepower'].mean()
std = dataframe['horsepower'].std()
dataframe['horsepower'] = (dataframe['horsepower'] - mean) / std

scaled_values_for_X0 = dataframe['X0'].values
scaled_values_for_horsepower = dataframe['horsepower'].values
target_values = dataframe['mpg'].values

W = [5, 3]  # initial weights
alpha = 0.1  # Learning rate
n = 398  # Number of examples
prev_cost = get_cost(scaled_values_for_X0, scaled_values_for_horsepower, target_values, n, W)
# Making infinity loop until the data converge

while True:
    gradients = []
    X0_summation = 0
    horsepower_summation = 0

    # Make a gradient
    for i in range(0, n):
        # Make the summation for horsepower and for X0
        f_x = W[0] * scaled_values_for_X0[i] + W[1] * scaled_values_for_horsepower[i]

        X0_summation = X0_summation + (f_x - target_values[i]) * scaled_values_for_X0[i]
        horsepower_summation = horsepower_summation + \
                               (f_x - target_values[i]) * scaled_values_for_horsepower[i]

    gradients.append((2 / n) * X0_summation)
    gradients.append((2 / n) * horsepower_summation)

    #Change the value of W to the new ones
    W[0] = W[0] - alpha * gradients[0]
    W[1] = W[1] - alpha * gradients[1]

    current_cost = get_cost(scaled_values_for_X0, scaled_values_for_horsepower, target_values, n, W)
    final_cost = abs(current_cost - prev_cost)

    if final_cost < 1e-12:
        break
    else:
        prev_cost = current_cost
print("\n\nBy the Gradient Descent -----> W[0] = {}, W[1] = {}".format(W[0], W[1]))

predicted_output = W[0] * dataframe['X0'] + W[1] * dataframe['horsepower']
# Plot the scatter plot
dataframe.plot(kind='scatter', x='horsepower', y='mpg', label='Horsepower Data')
# Plot the regression line
plt.plot(dataframe['horsepower'], predicted_output, color='red', label='Gradient Descent Algorithm')
plt.xlabel('Horsepower')
plt.ylabel('mpg')
plt.legend()
plt.show()
