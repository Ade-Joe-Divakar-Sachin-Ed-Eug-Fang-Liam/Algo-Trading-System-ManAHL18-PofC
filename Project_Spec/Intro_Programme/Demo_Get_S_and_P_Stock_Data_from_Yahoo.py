
# coding: utf-8

# ## Demo: Get S&P stock data from Yahoo or Quandl

# ### Define the imports

# In[4]:


#!pip install pandas-datareader
#!pip install quandl


# In[5]:


import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
import quandl


# ### Specify global variables

# In[6]:


data_path = 'Stocks_Data'
sp500_ticker_url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
ticker_cache_filename = 'sp500tickers.pickle'
ticker_cache_path = '{}/{}'.format(data_path, ticker_cache_filename)
my_quandl_api_key = 'sRoopcCR-jfCKimYgz65'
tickers_of_interest = ['GOOGL', 'AMZN', 'AAPL', 'ORCL', 'MSFT']


# ### Find,scrap and save the list of S&P 500 tickers
# - Tickers are scraped from wikipedia url: 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
# - Ticker is then pickled (cached)

# In[7]:


def save_sp500_tickers():
    resp = requests.get(sp500_ticker_url)
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open(ticker_cache_path,"wb") as f:
        pickle.dump(tickers,f)
    
    print('S&P 500 ticker have been sourced from: {0} and been cached here: {1}'.format(sp500_ticker_url, ticker_cache_path))
        
    return tickers


# In[8]:


#tickers = save_sp500_tickers()


# ### Get the S & P Index from 2000-01-01 to 2018-0-01

# In[35]:


def getSP500PriceIndecies(start_date = "2000-01-01", end_date = "2018-01-01"):
    #quandl.get("BCIP/_INX", authtoken="sRoopcCR-jfCKimYgz65", start_date=start_date, end_date=end_date)
    #return quandl.get("CME/SPU2017", authtoken="sRoopcCR-jfCKimYgz65", start_date=start_date, end_date=end_date)
    return quandl.get_table('WIKI/PRICES', date='1999-11-18', ticker='^GSPC')

sp500_data = getSP500PriceIndecies()


# In[36]:


sp500_data.head(10)


# ### Now we will use pandas_datareader api to get the stock data for each s&p 500 ticker fom Yahoo

# In[14]:


def getDataFromYahoo(reload_sp500=False, data_source='quandl'):
    
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open(ticker_cache_path,"rb") as f:
            tickers = pickle.load(f)
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2017, 12, 31)
    
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        ticker_cache_loc = '{0}/{1}.csv'.format(data_path, ticker)
        if not os.path.exists(ticker_cache_loc) and (ticker in tickers_of_interest):            
            print("Currently sourcing data for ticker: {0} from source: {1}..\n".format(ticker, data_source))
            if data_source == 'yahoo':
                # Use Yahoo
                df = web.DataReader(ticker, "yahoo", start, end)                
            else:
                # Use Quandl
                quandl.ApiConfig.api_key = my_quandl_api_key
                df = quandl.get("WIKI/{}".format(ticker), start_date=start, end_date=end)
            df.to_csv(ticker_cache_loc)            
            print("Souced data will be cached here: {}\n".format(ticker_cache_loc))
        else:
            print('Already have ticker: {} or it is not required!'.format(ticker))


# In[16]:


#getDataFromYahoo()


# ### Read the ticker data for each tech company into a Pandas dataframe 

# In[64]:


def createDataframePerTicker():
    tables = {}
    main_table = pd.DataFrame()
    for count,ticker in enumerate(tickers_of_interest):
        df = pd.read_csv('{0}/{1}.csv'.format(data_path, ticker))
        df.set_index('Date', inplace=True)
        
        # Compute daily Open/Close and High/Low percentage difference
        df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
        df['{}_OC_pct_diff'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']
        
        # Also rename the 'Adj. Close' column as ticker value
        df.rename(columns={'Adj. Close':ticker}, inplace=True)
        
        # Also rename the 'Adj. Volume' column as 'Adj_Volume'
        df.rename(columns={'Adj. Volume':'{}_Adj_Volume'.format(ticker)}, inplace=True)
        
        # Also remove the unwanted colums: ['Open','High','Low','Close', 'Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low']
        df.drop(['Open','High','Low','Close', 'Volume','Ex-Dividend', 'Split Ratio', 'Adj. Open', 'Adj. High', 'Adj. Low'],1,inplace=True)
        tables[ticker] = df
        
        # Join the individual ticker tables into a main table
        if main_table.empty:
            main_table = df
        else:
            main_table = main_table.join(df, how='outer')
    return main_table, tables
            


# In[65]:


#tables = createDataframePerTicker()
main_table, tables = createDataframePerTicker()


# ### Lag the returns of the stock price

# In[70]:


def lagStockReturns(tables, lag = 5):
    new_tables = {}
    for ticker, df in tables.items():
        return_col = '{}_OC_pct_diff'.format(ticker)
        lag_back_col = 'Lag_bwd_'
        lag_fwd_col = 'Lag_fwd_'
        volume_col = 'Adj_Volume'
        for i in range(1, lag, 1):
            df.loc[:,'{0}{1} '.format(lag_back_col, str(i))] = df[return_col].shift(i)
        new_df = df[[x for x in df.columns if lag_back_col in x or x == return_col or volume_col in x]].iloc[lag:,]
        new_tables[ticker] = new_df
    return new_tables

tables_new = lagStockReturns(tables, lag = 5)


# In[71]:


tables_new['GOOGL']


# In[53]:


#tables['GOOGL']
#main_table['2004-08-01':]
#main_table.index


# In[21]:


# Define which online source one should use
data_source = 'google'

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2000-01-01'
end_date = '2018-01-01'

tickers = ['SPY']

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = web.DataReader(tickers, data_source, start_date, end_date)

panel_data.head(5)

