# import necessary libraries
import pandas as pd
import yfinance as yf
import warnings
import os
from IPython.display import display


# define it a class for data cleaning
class data_cleaning:
    def __init__(self, option_data):
        self.option_data = option_data
        # self.expiry_list = None  # Initialize expiry_list as None

    def format_data(self, index='SPX'):
        # ignore SettingWithCopyWarning:
        pd.options.mode.chained_assignment = None
        
        # drop na rows
        self.option_data = self.option_data.dropna()
        
        # ticker breakdown for call
        self.option_data[['Index', 'Expiry', 'strike']] = self.option_data['Ticker'].str.split(' ', expand=True)
        self.option_data['Type'] = self.option_data['strike'].str[0]
        self.option_data.drop(['strike'], axis=1, inplace=True)

        # ticker breakdown for put
        self.option_data[['Index.1', 'Expiry.1', 'strike']] = self.option_data['Ticker.1'].str.split(' ', expand=True)
        self.option_data['Type.1'] = self.option_data['strike'].str[0]
        self.option_data.drop(['strike'], axis=1, inplace=True)

        # spit call and put data
        call_data = self.option_data[self.option_data.columns[~self.option_data.columns.str.contains('.1')]]
        put_data = self.option_data[self.option_data.columns[self.option_data.columns.str.contains('.1')]]
        put_data.columns = call_data.columns
        call_data = call_data[['Ticker', 'Index', 'Type', 'Expiry', 'Strike', 'Bid', 'Ask', 'Last', 'Volm', 'IVM']]
        put_data = put_data[['Ticker', 'Index', 'Type', 'Expiry', 'Strike', 'Bid', 'Ask', 'Last', 'Volm', 'IVM']]

        # redefine option data as call concat put, convert strike to integer, convert expiry to string format, sort data by expiray and strike, delete 0 volume rows 
        self.option_data = pd.concat([call_data, put_data], axis=0)
        self.option_data['Strike'] = self.option_data['Strike'].astype(int)
        self.option_data['Expiry'] = pd.to_datetime(self.option_data['Expiry'], format='%m/%d/%y')
        self.option_data['Expiry'] = self.option_data['Expiry'].dt.date
        self.option_data = self.option_data.sort_values(by=['Type', 'Expiry', 'Index', 'Strike'])
        self.option_data['Expiry'] = self.option_data['Expiry'].astype(str)
        self.option_data = self.option_data[self.option_data['Volm'] != 0]
        
        # if the bid or ask price lower or greater than 10% of the last price, delete the row
        self.option_data = self.option_data[(self.option_data['Bid'] > self.option_data['Last'] * 0.9)]
        self.option_data = self.option_data[(self.option_data['Ask'] < self.option_data['Last'] * 1.1)]

        # For spx option, only use SPX data
        if index == 'SPX':
            self.option_data = self.option_data[self.option_data['Index'] == 'SPX']
        elif index == 'SPXW':
            self.option_data = self.option_data[self.option_data['Index'] == 'SPXW']
        elif index == 'NKY':
            self.option_data = self.option_data[self.option_data['Index'] == 'NKY']
        elif index == 'HSI':
            self.option_data = self.option_data[self.option_data['Index'] == 'HSI']

        # set index as number
        self.option_data = self.option_data.reset_index(drop=True)

        return self.option_data


    def check_iv_number(self, drop_type=None, drop_threshold=-1):
        # keep expiry with enough implied volatility data
        self.expiry_list = self.option_data['Expiry'].unique()
        expiry_keep = []

        for expiry in self.expiry_list:
            # for each expiry, get the unique strike
            option_data_expiry = self.option_data[self.option_data['Expiry'] == expiry]
            option_strike = option_data_expiry['Strike'].unique()

            # if strike_keep has more than 10 unique strike, keep the expiry
            if drop_threshold == -1:
                drop_threshold = 10
            if drop_type == 'volume':
                if len(option_strike) > drop_threshold:
                    expiry_keep.append(expiry)
            elif drop_type == None:
                expiry_keep.append(expiry)

        # keep the expiry with enough implied volatility data
        self.option_data = self.option_data[self.option_data['Expiry'].isin(expiry_keep)]

        return self.option_data
    

    def get_hist(self, index, start, end):
        # check if the file exists
        if not os.path.exists(f'./Public/Data/Index/{index}_hist_{start}_{end}.csv'):
            if index == 'SPX':
                ticker = yf.Ticker('^GSPC')
            elif index == 'NKY':
                ticker = yf.Ticker('^N225')
            elif index == 'HSI':
                ticker = yf.Ticker('^HSI')
            hist = ticker.history(start=start, end=end)
            hist = hist.reset_index()
            hist = hist[['Date', 'Open']]
            hist['Date'] = pd.to_datetime(hist['Date'], format='%Y-%m-%d')
            hist['Date'] = hist['Date'].dt.date
            hist['Date'] = hist['Date'].astype(str)
            hist = hist.rename(columns={'Close':index})
            hist = hist.set_index('Date')
            index = index.lower()
            hist.to_csv(f'./Public/Data/Index/{index}_hist_{start}_{end}.csv')
        else:
            hist = pd.read_csv(f'./Public/Data/Index/{index}_hist_{start}_{end}.csv')
            hist = hist.set_index('Date')
        
        return hist
    
    
    def extract_option_price(self, px_type='mid'):
        # ignore warning
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # store call price and put price with same strike and expiry in the same row of a dataframe
        option_price = pd.DataFrame(columns=['Index', 'Strike', 'Expiry', 'c', 'p'])
        expiry_list = self.option_data['Expiry'].unique()
        index = self.option_data['Index'].unique()
        
        for expiry in expiry_list:
            # for each expiry
            option_data_expiry = self.option_data[self.option_data['Expiry'] == expiry]
            option_strike = option_data_expiry['Strike'].unique()

            for strike in option_strike:
                # for each strike
                option_data_strike = option_data_expiry[option_data_expiry['Strike'] == strike]
                
                # if call and put price exist, store the price in the dataframe
                if px_type == 'mid':
                    # put price and call price are the mid price of bid and ask
                    call_price = option_data_strike[option_data_strike['Type'] == 'C'][['Bid', 'Ask']].mean(axis=1)
                    put_price = option_data_strike[option_data_strike['Type'] == 'P'][['Bid', 'Ask']].mean(axis=1)
                elif px_type == 'bid':
                    call_price = option_data_strike[option_data_strike['Type'] == 'C']['Bid']
                    put_price = option_data_strike[option_data_strike['Type'] == 'P']['Bid']
                elif px_type == 'ask':
                    call_price = option_data_strike[option_data_strike['Type'] == 'C']['Ask']
                    put_price = option_data_strike[option_data_strike['Type'] == 'P']['Ask']
                elif px_type == 'last':
                    call_price = option_data_strike[option_data_strike['Type'] == 'C']['Last']
                    put_price = option_data_strike[option_data_strike['Type'] == 'P']['Last']

                if not call_price.empty:
                    call_price = call_price.values[0]
                if not put_price.empty:
                    put_price = put_price.values[0]

                temp_price = pd.DataFrame({'Index': index, 'Strike': strike, 'Expiry': expiry, 'c': call_price, 'p': put_price}, index=[0])
                option_price = pd.concat([option_price, temp_price], axis=0)
        
        # set index as number
        option_price = option_price.reset_index(drop=True)

        return option_price
    
    


    
    
# -------------------------- test code --------------------------
if __name__ == '__main__':
    option_data_spx = pd.read_csv('./Public/Data/Option/spx_option_0901.csv')
    option_data_spx = data_cleaning(option_data_spx).format_data(index = 'SPX')
    option_data_spx = data_cleaning(option_data_spx).check_iv_number(drop_type='volume', drop_threshold=10)
    spx_data = data_cleaning(option_data_spx).get_hist('SPX', '2021-09-01', '2023-11-13')
    option_price_spx = data_cleaning(option_data_spx).extract_option_price(px_type='mid')

    # concat three index
    option_data = pd.concat([option_data_spx, option_data_nky, option_data_hsi], axis=0)
    option_price = pd.concat([option_price_spx, option_price_nky, option_price_hsi], axis=0)
    