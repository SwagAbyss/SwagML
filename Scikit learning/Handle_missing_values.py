# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Importing Data Packages

# %%
# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno

# Data preprocessing
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


# %%
data = pd.read_csv("./Titanic-Dataset.csv")


# %%
data.head()


# %%
data.columns


# %%
data.info()


# %%
data.describe()

# %% [markdown]
# # 2. Data Preprocessing
# %% [markdown]
# ### 2.1. Handling Missing Values
#
# What is Missing Data?
#
#     Missing data means absence of observations in columns.
#     It appears in values such as “0”, “NA”, “NaN”, “NULL”, “Not Applicable”, “None”.
#
# Two ways to solve missing values. There are,
# - Deletion
# - Imputation

# %%
# Splitting into different data types
numcol = data.select_dtypes(include=["int64", "float64"]).columns
objcol = data.select_dtypes(include=["object"])


# %%
print("Missing values and their percentage: ")
for column in data.columns:
    if data[column].isnull().any():
        print(
            f"Column: {column} - Values: {data[column].isnull().sum()} - Percentage: {round((data[column].isnull().sum())/data.shape[0]*100)}% - Type: {data[column].dtype}"
        )

# %% [markdown]
# #### Using dropna function to remove missing value rows

# %%
# Deleting Row wise
data["Cabin"].dropna(
    inplace=True
)  # Don't use or use with caution(Too much of data lose)

# Deleting Column wise
data.drop(columns=["Cabin"], inplace=True)  # 77% of data were missing in Cabin column


# %%
data["Embarked"].fillna(value=data.Embarked.mode())

# %% [markdown]
# #### 2.1.1. Using Simple Imputer to handling missing values
# %% [markdown]
# Using Sklearn Imputers
#
# strategy = {'mean', 'constant', 'most_frequent', 'median'}

# %%
from sklearn.impute import SimpleImputer

impute_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
data[numcol] = impute_mean.fit_transform(data[numcol].values)


# %%
from sklearn.impute import SimpleImputer

impute_mean = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
data["Embarked"] = impute_mean.fit_transform(data[["Embarked"]])


# %%
# for only one column (Simple Imputer, KNNImputer)
data["Age"] = impute_mean.fit_transform(data[["Age"]])

# %% [markdown]
# #### 2.1.2. Using KNNImputer to handling missing values

# %%
from sklearn.impute import KNNImputer

impute_KNN = KNNImputer(missing_values=np.nan, n_neighbors=5)
data[numcol] = impute_KNN.fit_transform(data[numcol].values)


# %%
msno.bar(data)
