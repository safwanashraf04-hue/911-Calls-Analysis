#!/usr/bin/env python
# coding: utf-8

# # 911 Calls Capstone Project

# For this capstone project we will be analyzing some 911 call data. The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 
# Just go along with this notebook and try to complete the instructions or answer the questions in bold using your Python and Data Science skills!

# ## Data and Setup

# **Import numpy and pandas**

# In[1]:


import numpy as np
import pandas as pd


# **Import visualization libraries and set %matplotlib inline.**

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Read in the csv file as a dataframe called df**

# In[3]:


df=pd.read_csv('911.csv')


# **Check the info() of the df**

# In[4]:


df.info()


# **Check the head of df**

# In[5]:


df.head()


# ## Basic Questions

# **What are the top 5 zipcodes for 911 calls?**

# In[6]:


df1=df['zip'].value_counts()
df1.head()


# **What are the top 5 townships (twp) for 911 calls?**

# In[7]:


twp=df['twp'].value_counts()
twp.head()


# **Take a look at the 'title' column, how many unique title codes are there?**

# In[8]:


df['title'].nunique()


# ## Creating new features

# **In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS.**

# In[9]:


df['Reason']=df['title'].apply(lambda y: y.split(':')[0] )
df['Reason'].head()


# **What is the most common Reason for a 911 call based off of this new column?**

# In[10]:


reason=df['Reason'].value_counts()
reason.head(1)


# **Now use seaborn to create a countplot of 911 calls by Reason.**

# In[11]:


sns.countplot(x='Reason',data=df)
plt.show()


# **Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?**

# In[12]:


type(df['timeStamp'].iloc[0])


# **You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects.**

# In[13]:


df['timeStamp']=pd.to_datetime(df['timeStamp'])
df['timeStamp'].head()


# **You can now grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# **You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.**

# In[14]:


time = df['timeStamp'].iloc[0]
time.day


# In[15]:


df['Hour']=df['timeStamp'].apply(lambda time:time.hour)
df['Month']=df['timeStamp'].apply(lambda time:time.month)
df['Day']=df['timeStamp'].apply(lambda time:time.dayofweek)





# **Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:**
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[16]:


dmap = {0:'Mon',1:'Tues',2:'Wed',3:'Thur',4:'Fri',5:'Sat',6:'Sun'}
df['Day of week']=df['Day'].apply(lambda int:dmap[int])
df['Day of week'].head()


# **Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.**

# In[17]:


sns.countplot(x='Day of week',data=df,hue='Reason')
plt.show()


# **Now do the same for Month:**

# In[18]:


sns.countplot(data=df,x='Month',hue='Reason')
plt.show()


# **Did you notice something strange about the Plot?**

# **You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas...**

# **Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame.**

# In[19]:


byMonth = df.groupby('Month').count()


# In[20]:


byMonth


# **Now create a simple plot off of the dataframe indicating the count of calls per month.**

# In[22]:


plt.plot(byMonth['lat'])
plt.show()


# **Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column.**

# In[26]:


byMonth = byMonth.reset_index().rename({'lat':'count'},axis=1)


# In[27]:


byMonth


# In[28]:


sns.lmplot(data=byMonth,x='Month',y='count')
plt.show()


# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method.** 

# In[34]:


df['Date'] = df['timeStamp'].apply(lambda time:time.date())
df['Date'].head()


# **Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[35]:


byDate = df.groupby('Date').count()


# In[36]:


byDate.name='count'


# In[39]:


plt.figure(figsize=(10,6))
plt.plot(byDate['lat'])
plt.show()


# **Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[40]:


reason_byDate = df.groupby(['Reason','Date']).count()


# In[41]:


reason_byDate.name = 'count'


# In[42]:


reason_byDate = reason_byDate.reset_index()


# In[58]:


plt.figure(figsize=(10,6))
df[df['Reason']=='Traffic'].groupby(by='Date').count()['lat'].plot()
plt.title('Traffic')
plt.show()


# In[59]:


plt.figure(figsize=(10,6))
df[df['Reason']=='Fire'].groupby(by='Date').count()['lat'].plot()
plt.title('Fire')
plt.show()


# In[60]:


plt.figure(figsize=(10,6))
df[df['Reason']=='EMS'].groupby(by='Date').count()['lat'].plot()
plt.title('EMS')
plt.show()


# **Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# In[64]:


dfData=df.groupby(['Day of week','Hour']).count()['lat'].unstack()
dfData=dfData.loc[['Sun','Mon','Tues','Wed','Thur','Fri','Sat']]
dfData


# **Now create a HeatMap using this new DataFrame.**

# In[67]:


plt.figure(figsize=(12,6))
sns.heatmap(data=dfData,cmap='viridis')
plt.show()


# **Now create a clustermap using this DataFrame.**

# In[68]:


plt.figure(figsize=(12,6))
sns.clustermap(data=dfData,cmap='viridis')
plt.show()


# **Now repeat these same plots and operations, for a DataFrame that shows the Month as the column.**

# In[71]:


dfData1=df.groupby(['Day of week','Month']).count()['lat'].unstack()
dfData1=dfData1.loc[['Sun','Mon','Tues','Wed','Thur','Fri','Sat']]
dfData1


# In[82]:


plt.figure(figsize=(10,5))
sns.heatmap(data=dfData1,cmap='viridis')
plt.show()


# In[83]:


plt.figure(figsize=(10,5))
sns.clustermap(data=dfData1,cmap='viridis')
plt.show()


# 
# # Great Job!
