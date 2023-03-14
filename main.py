#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Required Libraries..
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:


# Download Bangluru Housing DataSet from kaggle then load dataset ussing pandas..
df = pd.read_csv(r"C:\Users\HP\Desktop\CSV file\Bengaluru_House_Data.csv")
df.head()

# In[3]:


# Shape of Dataset(Number of Raws and Columns)
df.shape

# In[4]:


# A short Information of Dataset..
df.info()

# In[5]:


# Missing values in Dataset..
df.isna().sum().sort_values(ascending=False)

# In[6]:


# Visualize missing data..
sns.heatmap(df.isna(), cbar=False, cmap='summer')

# ## Handle Missing Values:
#

# In[7]:


# Too many missing values in column 'Society',society column has to be dropped,Also droping Availabilty column from the dataset..
df = df.drop(columns=['society', 'availability'])

# In[8]:


print(df['size'].mode())
print(df['location'].mode())

# In[9]:


# fill missing values columns
df['size'] = df['size'].fillna('2 bhk')
df['balcony'] = df['balcony'].fillna(df['balcony'].median())
df['bath'] = df['bath'].fillna(df['bath'].median())
df['location'] = df['location'].fillna('Whitefield')

# In[10]:


df.isna().sum()

# In[11]:


# Describe dataset..
df.describe()

# ### we can see some outliers in bathroom and price columns, Now let's look at the dataset a little better..

# In[12]:


for column in df.columns:
    print(df[column].value_counts())
    print('*' * 30)

# In[13]:


df['size'].value_counts()

# In[14]:


# Get the bhk in order..
df['bhk'] = df['size'].str.split().str.get(0).astype(int)

# In[15]:


df['bhk'].value_counts()

# In[16]:


df[df.bhk > 20]

# In[17]:


# now we can Drop size column..
df = df.drop('size', axis=1)

# In[18]:


# now split total_sqft and convert into numerical value..
df['total_sqft'].unique()


# In[19]:


def sqft_to_num(x):
    value = x.split('-')

    if len(value) == 2:
        return (float(value[0]) + float(value[1])) / 2

    try:
        return float(x)
    except:
        return None


# In[20]:


df['total_sqft'] = df['total_sqft'].apply(sqft_to_num)

# In[21]:


df.isna().sum()

# In[22]:


sns.heatmap(df.isna(), cbar=False)

# In[23]:


df['total_sqft'] = df['total_sqft'].fillna(df['total_sqft'].mean())

# In[24]:


df.isna().sum()

# In[25]:


# create new column as price per sqft..
df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']

# In[26]:


df['price_per_sqft']

# In[27]:


df['location'] = df['location'].apply(lambda x: x.strip())

# In[28]:


location_count = df['location'].value_counts()
location_count

# location_counts, if it is less than 10, then we will assign it to "other" categories.'''

# In[29]:


location_count_less_10 = location_count[location_count < 10]
location_count_less_10

# In[30]:


df['location'] = df['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)
df['location'].value_counts()

# In[31]:


df.describe()

# # Remove Outlier from dataset..

# In[32]:


# first removing outliers from total_sqft..
df_new = df[~(df['total_sqft'] / df['bhk'] < 300)]

# In[33]:


df_new.describe()

# In[34]:


df_new.shape


# In[35]:


def remove_outliers(data, column_name):
    # Calculate the first and third quartiles (Q1 and Q3)
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    # Calculate the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filter the data to remove values outside the bounds
    filtered_data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    return filtered_data


# In[36]:


df1 = remove_outliers(df_new, 'price_per_sqft')

# In[37]:


df1.describe()

# In[38]:


df1.shape


# In[39]:


def remove_bhk_outliers(data):
    exclude_indices = []
    for location, location_df in data.groupby('location'):
        for bhk in location_df['bhk'].unique():
            # Calculate the mean and standard deviation of price_per_sqft for each bhk in the location
            bhk_stats = location_df[location_df['bhk'] == bhk]['price_per_sqft'].describe()
            mean, std = bhk_stats['mean'], bhk_stats['std']
            # Filter out the rows where price_per_sqft is outside 1 standard deviation from the mean
            exclude_indices += location_df[(location_df['bhk'] == bhk) & (
                        (location_df['price_per_sqft'] < mean - std) | (
                            location_df['price_per_sqft'] > mean + std))].index.tolist()
    # Drop the excluded rows from the DataFrame and return the filtered DataFrame
    return data.drop(exclude_indices, axis='index')


# In[40]:


df2 = remove_bhk_outliers(df1)

# In[41]:


df2.shape

# In[42]:


df2.describe()

# In[43]:


df2.head()

# In[44]:


df_final = df2.drop(['area_type', 'price_per_sqft', 'balcony'], axis='columns')

# In[45]:


df_final.to_csv('cleaned_data.csv')

# In[46]:


[df_final['location'] != 'Cunningham Road']

# In[47]:


df_final.head()

# In[48]:


# select dependent and independet data from df_final..
X = df_final.drop(columns=['price'])
y = df_final['price']

# In[49]:


print(X.shape, y.shape)

# In[50]:


# import required libraries for model..
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# In[51]:


# Split dataset into training and test datasets..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# In[52]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# In[53]:


column_trans = make_column_transformer((OneHotEncoder(sparse_output=False), ['location']), remainder='passthrough')

# In[54]:


scaler = StandardScaler()

# ## Linear Regression

# In[55]:


lr = LinearRegression()

# In[56]:


pipe = make_pipeline(column_trans, scaler, lr)

# In[57]:


pipe.fit(X_train, y_train)

# In[58]:


y_pred_lr = pipe.predict(X_test)

# In[59]:


score = r2_score(y_test, y_pred_lr)
score

# ## Lasso

# In[60]:


ls = Lasso()
# parameters = {'alpha': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,5,6,7,8,9,10,15]}
# lassocv = GridSearchCV(ls, parameters, scoring = 'neg_mean_squared_error', cv=10)


# In[61]:


pipe = make_pipeline(column_trans, scaler, ls)

# In[62]:


pipe.fit(X_train, y_train)

# In[63]:


y_pred_ls = pipe.predict(X_test)

# In[64]:


score_ls = r2_score(y_test, y_pred_ls)
score_ls

# ## Ridge

# In[65]:


rid = Ridge()

# In[66]:


# parameters = {'alpha': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,5,10,20,30,40,50,60,70,80,85,90,95,100]}
# ridgecv = GridSearchCV(rid, parameters, scoring = 'neg_mean_absolute_error', cv=5,n_jobs=-1)


# In[67]:


pipe = make_pipeline(column_trans, scaler, rid)

# In[68]:


pipe.fit(X_train, y_train)

# In[69]:


y_pred_rid = pipe.predict(X_test)

# In[70]:


score_rid = r2_score(y_test, y_pred_rid)

# In[71]:


score_rid

# In[72]:


print("Linear Regression: ", score)
print("Lasso:             ", score_ls)
print("Ridge:             ", score_rid)

# In[73]:


import pickle

# In[77]:


pickle.dump(pipe, open('Ridgemodel.pkl', 'wb'))





