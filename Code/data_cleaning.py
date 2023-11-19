# import necessary libraries
import pandas as pd
import yfinance as yf
import warnings
import os


# define it a class for data cleaning
class data_cleaning:
    def __init__(self, option_data):
        self.option_data = option_data
        # self.expiry_list = None  # Initialize expiry_list as None

    def format_data(self):
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

        # only use SPX data
        self.option_data = self.option_data[self.option_data['Index'] == 'SPX']

        # set index as number
        self.option_data = self.option_data.reset_index(drop=True)

        return self.option_data


    def check_iv_number(self):
        # keep expiry with enough implied volatility data
        self.expiry_list = self.option_data['Expiry'].unique()
        expiry_keep = []

        for expiry in self.expiry_list:
            # for each expiry, get the unique strike
            option_data_expiry = self.option_data[self.option_data['Expiry'] == expiry]
            option_strike = option_data_expiry['Strike'].unique()

            # if strike_keep has more than 10 unique strike, keep the expiry
            if len(option_strike) > 10:
                expiry_keep.append(expiry)

        # keep the expiry with enough implied volatility data
        self.option_data = self.option_data[self.option_data['Expiry'].isin(expiry_keep)]

        return self.option_data
    

    def get_spx_hist(self, start, end):
        # check if the file exists
        if not os.path.exists(f'./Public/data/index_price/spx_hist_{start}_{end}.csv'):
            spx = yf.Ticker('^GSPC')
            spx_hist = spx.history(start=start, end=end)
            spx_hist = spx_hist.reset_index()
            spx_hist = spx_hist[['Date', 'Open']]
            spx_hist['Date'] = pd.to_datetime(spx_hist['Date'], format='%Y-%m-%d')
            spx_hist['Date'] = spx_hist['Date'].dt.date
            spx_hist['Date'] = spx_hist['Date'].astype(str)
            spx_hist = spx_hist.rename(columns={'Close':'SPX'})
            spx_hist = spx_hist.set_index('Date')
            spx_hist.to_csv(f'./Public/data/index_price/spx_hist_{start}_{end}.csv')
        else:
            spx_hist = pd.read_csv(f'./Public/data/index_price/spx_hist_{start}_{end}.csv')
            spx_hist = spx_hist.set_index('Date')
        
        return spx_hist
    
    
    def extract_option_price(self):
        # ignore warning
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # store call price and put price with same strike and expiry in the same row of a dataframe
        option_price = pd.DataFrame(columns=['Strike', 'Expiry', 'c', 'p'])
        option_data_spx = self.option_data[self.option_data['Index'] == 'SPX']
        expiry_list = option_data_spx['Expiry'].unique()
        
        for expiry in expiry_list:
            # for each expiry
            option_data_expiry = self.option_data[self.option_data['Expiry'] == expiry]
            option_strike = option_data_expiry['Strike'].unique()

            for strike in option_strike:
                # for each strike
                option_data_strike = option_data_expiry[option_data_expiry['Strike'] == strike]
                
                # if call and put price exist, store the price in the dataframe
                # call_price = option_data_strike[option_data_strike['Type'] == 'C']['Last']
                # put_price = option_data_strike[option_data_strike['Type'] == 'P']['Last']
                # put price and call price are the mid price of bid and ask
                call_price = option_data_strike[option_data_strike['Type'] == 'C'][['Bid', 'Ask']].mean(axis=1)
                put_price = option_data_strike[option_data_strike['Type'] == 'P'][['Bid', 'Ask']].mean(axis=1)

                if not call_price.empty:
                    call_price = call_price.values[0]
                if not put_price.empty:
                    put_price = put_price.values[0]

                temp_price = pd.DataFrame({'Strike': strike, 'Expiry': expiry, 'c': call_price, 'p': put_price}, index=[0])
                option_price = pd.concat([option_price, temp_price], axis=0)
        
        # set index as number
        option_price = option_price.reset_index(drop=True)

        return option_price
    
    


    
    
# -------------------------- test code --------------------------
if __name__ == '__main__':
    raw_data = pd.read_csv('./Public/data/option_price/20230901/spx_option_0901.csv')
    option_data = data_cleaning(raw_data).format_data()
    print(option_data)
    # spx_hist = get_spx_hist('2021-08-01', '2021-09-01')
    # print(spx_hist)