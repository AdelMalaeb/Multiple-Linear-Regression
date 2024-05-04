#!/usr/bin/env python
# coding: utf-8


# Imports
# Packages for numerics + dataframes
import pandas as pd
import numpy as np

# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for date conversions for calculating trip durations
from datetime import datetime
from datetime import date
from datetime import timedelta

# Packages for OLS, MLR, confusion matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics # For confusion matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error




df0=pd.read_csv("/Users/adel/Desktop/TLC/2017_Yellow_Taxi_Trip_Data.csv")




#Create a copy of the data frame and keep the original untouched "Preserving the original data frame"
df = df0.copy()
df.head()


#Get the basic information of the data types**
df.info()

# Check for Missing Values
print('Missing values per column:')
df.isna().sum()


#Check for Duplicates
print('Shape of dataframe:', df.shape)
print('Shape of dataframe with duplicates dropped:', df.drop_duplicates().shape)



#Descriptive Statistics
df.describe()

# Check the format of the data
df['tpep_dropoff_datetime'][6]


# Convert `tpep_pickup_datetime` to datetime format
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# Convert `tpep_dropoff_datetime` to datetime format
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# Display data types of `tpep_pickup_datetime`, `tpep_dropoff_datetime`
print('Data type of tpep_pickup_datetime:', df['tpep_pickup_datetime'].dtype)
print('Data type of tpep_dropoff_datetime:', df['tpep_dropoff_datetime'].dtype)



# Create `duration` column
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])/np.timedelta64(1,'m')


# **np.timedelta64(1, 'm'):** This part of the code divides each timedelta by a timedelta representing one minute (np.timedelta64(1, 'm')). This converts the time difference from the default units (likely nanoseconds) to minutes.

# #### Create a Rush hour Column
# 
# Define rush hour as:
# * Any weekday (not Saturday or Sunday) AND
# * Either from 06:00&ndash;10:00 or from 16:00&ndash;20:00
# 
# Create a binary `rush_hour` column that contains a 1 if the ride was during rush hour and a 0 if it was not.

# For that, I have to create a **day** and **month** columns

# Create 'day' col
df['day'] = df['tpep_pickup_datetime'].dt.day_name().str.lower()

# Create 'month' col
df['month'] = df['tpep_pickup_datetime'].dt.strftime('%b').str.lower()


# Create 'rush_hour' col
df['rush_hour'] = df['tpep_pickup_datetime'].dt.hour

# If day is Saturday or Sunday, impute 0 in `rush_hour` column
df.loc[df['day'].isin(['saturday', 'sunday']), 'rush_hour'] = 0


def rush_hour_computation(hour):
    if 6 <= hour['rush_hour'] < 10:
        val = 1
    elif 16 <= hour['rush_hour'] < 20:
        val = 1
    else:
        val = 0
    return val

# Apply the `rush_hour()` function to the new column
df.loc[(df.day != 'saturday') & (df.day != 'sunday'), 'rush_hour'] = df.apply(rush_hour_computation, axis=1)
df.head()


# Check for Outliers

fig, axes = plt.subplots(1, 3, figsize=(15, 2))
fig.suptitle('Boxplots for outlier detection')
sns.boxplot(ax=axes[0], x=df['trip_distance'])
sns.boxplot(ax=axes[1], x=df['fare_amount'])
sns.boxplot(ax=axes[2], x=df['duration'])

# Save the figure before displaying it
plt.savefig("Boxplots for outlier detection1.pdf")

plt.show();



df['trip_distance'].describe()

df['fare_amount'].describe()



IQR_fare_amount = df["fare_amount"].quantile(0.75) - df["fare_amount"].quantile(0.25)
print("The IQR of fare amount is:" , IQR_fare_amount)


# Impute values less than $0 with 0
df.loc[df['fare_amount'] < 0, 'fare_amount'] = 0
df['fare_amount'].min()


# Now impute the maximum value as `Q3 + (6 * IQR)`


def outlier_imputer(column_list, iqr_factor):
    '''
    Impute upper-limit values in specified columns based on their interquartile range.

    Arguments:
        column_list: A list of columns to iterate over
        iqr_factor: A number representing x in the formula:
                    Q3 + (x * IQR). Used to determine maximum threshold,
                    beyond which a point is considered an outlier.

    The IQR is computed for each column in column_list and values exceeding
    the upper threshold for each column are imputed with the upper threshold value.
    '''
    for col in column_list:
        # Reassign minimum to zero
        df.loc[df[col] < 0, col] = 0

        # Calculate upper threshold
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_threshold = q3 + (iqr_factor * iqr)
        print(col)
        print('q3:', q3)
        print('upper_threshold:', upper_threshold)

        # Reassign values > threshold to threshold
        df.loc[df[col] > upper_threshold, col] = upper_threshold
        print(df[col].describe())
        print()


outlier_imputer(['fare_amount'], 6)


#Dealing with duration outliers
df['duration'].describe()

# Impute a 0 for any negative values
df.loc[df['duration'] < 0, 'duration'] = 0
df['duration'].min()

# Impute the high outliers
outlier_imputer(['duration'], 6)


# Feature Engineering / Transformations

# Create `pickup_dropoff` column
df['pickup_dropoff'] = df['PULocationID'].astype(str) + ' ' + df['DOLocationID'].astype(str)
df['pickup_dropoff'].head(2)


df.head()


# Now, I use a `groupby()` statement to group each row by the new `pickup_dropoff` column, compute the mean, and capture the values only in the `trip_distance` column. Assign the results to a variable named `grouped`.

grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
grouped[:5]


# 1. Convert `grouped` to a dictionary
grouped_dict = grouped.to_dict()

# 2. Reassign to only contain the inner dictionary
grouped_dict = grouped_dict['trip_distance']


# 1. Create a mean_distance column that is a copy of the pickup_dropoff  column
df['mean_distance'] = df['pickup_dropoff']

# 2. Map `grouped_dict` to the `mean_distance` column
df['mean_distance'] = df['mean_distance'].map(grouped_dict)

# Confirm that it worked
df[(df['PULocationID']==4) & (df['DOLocationID']==112)][['mean_distance']]


# Create mean_duration column
 
# Repeat the same process
grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['duration']]
grouped

# Create a dictionary where keys are unique pickup_dropoffs and values are
# mean trip duration for all trips with those pickup_dropoff combos
grouped_dict = grouped.to_dict()
grouped_dict = grouped_dict['duration']

df['mean_duration'] = df['pickup_dropoff']
df['mean_duration'] = df['mean_duration'].map(grouped_dict)

# Confirm that it worked
df[(df['PULocationID']==4) & (df['DOLocationID']==112)][['mean_duration']]


# ### Extract Relevant Feautres

df1 = df.copy()

df1 = df1[["fare_amount","mean_distance","mean_duration","rush_hour"]]


# Visualize the relationship between the variables
# Create the pair plot
sns.pairplot(df1[['fare_amount', 'mean_duration', 'mean_distance']])

# Display the figure
plt.show()


# Remove the target column from the features
X = df1.drop(columns=['fare_amount'])

# Set y variable
y = df1[['fare_amount']]

# Display first few rows
X.head()


# ### Split data into training and test sets
# Create training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ### Standardize the data

# Standardize the X variables
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print('X_train scaled:', X_train_scaled)


# ### Fit the model

# Fit your model to the training data
lr=LinearRegression()
lr.fit(X_train_scaled, y_train)


# ## Evaluate model

# Evaluate the model performance on the training data
r_sq = lr.score(X_train_scaled, y_train)
print('Coefficient of determination:', r_sq)

y_pred_train = lr.predict(X_train_scaled)



r2 = r_sq
MAE =  mean_absolute_error(y_train, y_pred_train)
MSE = mean_squared_error(y_train, y_pred_train)
RMSE = np.sqrt(mean_squared_error(y_train, y_pred_train))

#Create a metric dictionary
metric_dict = {"R2":r2,
               "MAE":MAE,
               "MSE": MSE,
               "RMSE": RMSE}

Train_results = pd.DataFrame(metric_dict,index = [0])
Train_results["Results"] = "Train Data"
Train_results


# Test data

# Scale the X_test data
X_test_scaled = scaler.transform(X_test)


# Evaluate the model performance on the test data
r_sq = lr.score(X_test_scaled, y_test)
print('Coefficient of determination:', r_sq)

y_pred_test = lr.predict(X_test_scaled)



r2 = r_sq
MAE =  mean_absolute_error(y_test, y_pred_test)
MSE = mean_squared_error(y_test, y_pred_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))

#Create a metric dictionary
metric_dict = {"R2":r2,
               "MAE":MAE,
               "MSE": MSE,
               "RMSE": RMSE}

Test_results = pd.DataFrame(metric_dict,index = [0])
Test_results["Results"] = "Test Data"
Test_results


# ## Prediction Results

# Create a `results` dataframe
results = pd.DataFrame(data={'actual': y_test['fare_amount'],
                             'predicted': y_pred_test.ravel()}) #converting it into a 1-dimensional array.
results['residual'] = results['actual'] - results['predicted']
results.head()

results.to_csv('Prediction Results.csv', index=False)  # Save as CSV without index


# ## Visualize model results
# Create a scatterplot to visualize `predicted` over `actual`
fig, ax = plt.subplots(figsize=(6, 6))
sns.set(style='whitegrid')
sns.scatterplot(x='actual',
                y='predicted',
                data=results,
                s=20,
                alpha=0.5,
                ax=ax
)
# Draw an x=y line to show what the results would be if the model were perfect

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--',linewidth=2)

plt.savefig("Actual vs. predicted.pdf")
plt.title('Actual vs. predicted');


# ## Check model Assumptions

# Visualize the distribution of the `residuals`
sns.histplot(results['residual'], bins=np.arange(-15,15.5,0.5))
plt.title('Distribution of the residuals')
plt.xlabel('residual value')
plt.ylabel('count')

plt.savefig('Distribution of the residuals.pdf')


# A normal distribution around zero is good, as it demonstrates that the model's errors are evenly distributed and unbiased.

# **Homoscedasticity**

# Create a scatterplot of `residuals` over `predicted`

sns.scatterplot(x='predicted', y='residual', data=results)
plt.axhline(0, c='red')
plt.title('Scatterplot of residuals over predicted values')
plt.xlabel('predicted value')
plt.ylabel('residual value')

plt.savefig('Homoscedasticity Assumption.pdf')
plt.show()



