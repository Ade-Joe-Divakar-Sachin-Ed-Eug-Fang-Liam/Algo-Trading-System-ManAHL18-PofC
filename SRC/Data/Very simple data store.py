
# coding: utf-8

# In[113]:


import datetime as dt
import pandas as pd
import pickle
import os
import datetime


# In[128]:


def getStoredData(srtdt, enddt, ticker):
    #currently assumes that csv data is organised in format: Date,Open,High,Low,Close,Adj Close,Volume
    #also assumes that the name of the csv is the same as that as the ticker
    path = r'C:\Users\Edward Stables\Documents\Programming\Jupyter\Man AHL\Data\Initial Datasets'
    beginning_of_time='01/01/2000'
    os.chdir(path)
    try:
        print("try")
        #attempt to load the csv file from the path directory
        frame = pd.read_csv(ticker+'.csv')
    except FileNotFoundError:
        #if the file doesn't exist then send a request to get the data for the ticker's dates
        frame = getdata(beginning_of_time, datetime.date.today().strftime('%d/%m/%Y'), ticker)
        frame.to_csv(ticker+'.csv')
    return frame.loc[srtdt:enddt]
    


# In[129]:


def getdata(startime, endtime, ticker):
    d=[123, 456, 789]
    df = pd.DataFrame(data=d)
    return df


# In[132]:


print(getStoredData('15-11-2011', '31-1-2012', '^GSPC'))

