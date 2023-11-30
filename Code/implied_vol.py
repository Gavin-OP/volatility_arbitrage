# import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from functools import partial
from IPython.display import display


# define BS model
def BS_fwd_pricer(F, K, T, sig, r, isCall):
    d1 = (np.log(F / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    
    if isCall:
        option_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        option_price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    return option_price


# use bisection method to find implied volatility
def bisection_implied_vol(F, K, T, r, option_price, tol=0.05, isCall=True):
    # leave the isCall and sig as variables for the pricer
    implied_vol_pricer = partial(BS_fwd_pricer, F = F, K = K, r=r, T=T)

    # define the BS - p calculator
    def diff_calculator(isCall, sig):
        temp_price = implied_vol_pricer(isCall=isCall, sig=sig)
        price_diff = temp_price - option_price
        return price_diff

    # initialize the value of the range of sig, and calculate the option price based on historical volatility
    upper_bound = 1
    lower_bound = 0.01
    temp_sig = 0.5
    price_diff = 1

    # find two bounds of the root
    while diff_calculator(isCall=isCall, sig=lower_bound) > 0:
        lower_bound = lower_bound ** 2

    upper_bound = lower_bound
    while diff_calculator(isCall=isCall, sig=upper_bound) < 0:
        upper_bound = upper_bound * 1.2

    # bisection algo
    while abs(price_diff) >= tol:
        temp_sig = (lower_bound + upper_bound) / 2
        price_diff = diff_calculator(isCall=isCall, sig=temp_sig)

        if price_diff > 0:
            upper_bound = temp_sig
        else:
            lower_bound = temp_sig

    return temp_sig


class BS_implied_vol():
    def __init__(self, option_price):
        self.option_price = option_price
        self.today = pd.to_datetime('2023-09-01', format='%Y-%m-%d')
    
    def parity_implied_params(self, option_data, plot_parity = False):
        # initialization
        implied_params = pd.DataFrame(columns=['Index', 'Expiry', 'implied_ir', 'implied_fwd'])
        index_list = option_data['Index'].unique()

        for index in index_list:
            expiry_list_index = option_data[option_data['Index'] == index]['Expiry'].unique()
            for expiry in expiry_list_index:
                # for each expiry
                option_price_expiry = self.option_price[self.option_price['Expiry'] == expiry]
                option_price_expiry = option_price_expiry.dropna()
                option_price_expiry['c-p'] = option_price_expiry['c'] - option_price_expiry['p']

                # regress c-p on -strike
                model = LinearRegression()
                x = option_price_expiry[['Strike']]
                y = option_price_expiry['c-p']
                if x.empty or y.empty:
                    continue
                T = (pd.to_datetime(expiry, format='%Y-%m-%d') - self.today).days / 365
                model.fit(x, y)
                discount_factor = -model.coef_[0]
                
                # get implied risk free rate and implied forward price
                if discount_factor == 0:
                    # use the previous expiry's interest rate and fwd price with modification
                    # location = np.where(expiry_list_index == expiry)[0][0]
                    # previous_expiry = expiry_list_index[location - 1]
                    # risk_free_rate = implied_params[implied_params['Expiry'] == previous_expiry]['implied_ir']
                    # forward_price = implied_params[implied_params['Expiry'] == previous_expiry]['implied_fwd']
                    # time_diff = (pd.to_datetime(expiry, format='%Y-%m-%d') - pd.to_datetime(previous_expiry, format='%Y-%m-%d')).days / 365
                    # forward_price = forward_price * np.exp(risk_free_rate * time_diff)
                    continue
                else:
                    forward_price = model.intercept_ / discount_factor
                    risk_free_rate = -np.log(discount_factor) / T

                # format the output
                implied_params = pd.concat([implied_params, pd.DataFrame({'Index': index, 'Expiry': expiry, 'implied_ir': risk_free_rate, 'implied_fwd': forward_price}, index=[0])], axis=0)
                
                # plot c-p vs strike
                if plot_parity:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(option_price_expiry['Strike'], option_price_expiry['c-p'], color='#0078d4')
                    plt.plot(option_price_expiry['Strike'], model.predict(x), color='#4a8cff')
                    plt.text(0.6, 0.8, f'c-p = {model.coef_[0]:.4f} * strike + {model.intercept_:.4f}', transform=plt.gca().transAxes, color='#a6a6a6')
                    plt.xlabel('Strike')
                    plt.ylabel('Call - Put')
                    title = index +': Put Call parity ' + expiry
                    title = ''.join(title)
                    plt.title(title)
                    # make a color of legend, axis, ticks, and formulas both suitable for dark and light background
                    plt.gca().spines['bottom'].set_color('#a6a6a6')
                    plt.gca().spines['top'].set_color('#a6a6a6')
                    plt.gca().spines['right'].set_color('#a6a6a6')
                    plt.gca().spines['left'].set_color('#a6a6a6')
                    plt.gca().tick_params(axis='x', colors='#a6a6a6')
                    plt.gca().tick_params(axis='y', colors='#a6a6a6')
                    plt.gca().yaxis.label.set_color('#a6a6a6')
                    plt.gca().xaxis.label.set_color('#a6a6a6')
                    plt.gca().title.set_color('#a6a6a6')
                    # save the plot if not exist
                    name = index + '_' + expiry
                    name = ''.join(name)
                    if not os.path.exists(f'./Public/Plot/Put_call_parity/{name}.png'):
                        plt.savefig(f'./Public/Plot/Put_call_parity/{name}.png', transparent=True)
                    plt.show()
            
            # add expiry that cannot be found by put call parity
            expiry_list_index_exist = implied_params[implied_params['Index'] == index]['Expiry'].unique()
            expiry_list_index_not_exist = expiry_list_index[~np.isin(expiry_list_index, expiry_list_index_exist)]
            expiry_list_index_exist = pd.to_datetime(expiry_list_index_exist, format='%Y-%m-%d')
            expiry_list_index_exist = np.sort(expiry_list_index_exist)
            for expiry in expiry_list_index_not_exist:
                if (expiry_list_index_exist < pd.to_datetime(expiry, format='%Y-%m-%d')).sum() == 0:
                    previous_expiry = expiry_list_index_exist[expiry_list_index_exist > pd.to_datetime(expiry, format='%Y-%m-%d')][-1]
                    previous_expiry = pd.to_datetime(previous_expiry, format='%Y-%m-%d').strftime('%Y-%m-%d')
                    while implied_params[(implied_params['Index'] == index) & (implied_params['Expiry'] == previous_expiry)]['implied_ir'].values[0] == 0:
                        previous_expiry = expiry_list_index_exist[expiry_list_index_exist > pd.to_datetime(previous_expiry, format='%Y-%m-%d')][-1]
                        previous_expiry = pd.to_datetime(previous_expiry, format='%Y-%m-%d').strftime('%Y-%m-%d')
                else:
                    previous_expiry = expiry_list_index_exist[expiry_list_index_exist < pd.to_datetime(expiry, format='%Y-%m-%d')][-1]
                    previous_expiry = pd.to_datetime(previous_expiry, format='%Y-%m-%d').strftime('%Y-%m-%d')
                    while implied_params[(implied_params['Index'] == index) & (implied_params['Expiry'] == previous_expiry)]['implied_ir'].values[0] < 0:
                        previous_expiry = expiry_list_index_exist[expiry_list_index_exist < pd.to_datetime(previous_expiry, format='%Y-%m-%d')][-1]
                        previous_expiry = pd.to_datetime(previous_expiry, format='%Y-%m-%d').strftime('%Y-%m-%d')
                # change the format of previous_expiry to string with YYYY-MM-DD
                previous_expiry = pd.to_datetime(previous_expiry, format='%Y-%m-%d').strftime('%Y-%m-%d')
                implied_ir = implied_params[(implied_params['Index'] == index) & (implied_params['Expiry'] == previous_expiry)]['implied_ir']
                implied_fwd = implied_params[(implied_params['Index'] == index) & (implied_params['Expiry'] == previous_expiry)]['implied_fwd']
                implied_params = pd.concat([implied_params, pd.DataFrame({'Index': index, 'Expiry': expiry, 'implied_ir': implied_ir, 'implied_fwd': implied_fwd}, index=[0])], axis=0)
                
        # reset index
        implied_params = implied_params.reset_index(drop=True)

        return implied_params

    # plot implied ir vs expiry
    def plot_ir(self, implied_params):
        index = implied_params['Index'].unique()
        plt.figure(figsize=(10, 6))
        plt.plot(implied_params['Expiry'], implied_params['implied_ir'], color='#4a8cff')
        plt.ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.xlabel('Expiry')
        plt.ylabel('Implied Interest Rate')
        title = index +': Implied Interest Rate vs Expiry'
        title = ''.join(title)
        plt.title(title)
        plt.gca().spines['bottom'].set_color('#a6a6a6')
        plt.gca().spines['top'].set_color('#a6a6a6')
        plt.gca().spines['right'].set_color('#a6a6a6')
        plt.gca().spines['left'].set_color('#a6a6a6')
        plt.gca().tick_params(axis='x', colors='#a6a6a6')
        plt.gca().tick_params(axis='y', colors='#a6a6a6')
        plt.gca().yaxis.label.set_color('#a6a6a6')
        plt.gca().xaxis.label.set_color('#a6a6a6')
        plt.gca().title.set_color('#a6a6a6')
        # save the plot if not exist
        
        plt.show()
    
    # plot implied fwd vs expiry
    def plot_fwd(self, implied_params):
        index = implied_params['Index'].unique()
        plt.figure(figsize=(10, 6))
        plt.plot(implied_params['Expiry'], implied_params['implied_fwd'], color='#4a8cff')
        plt.xticks(rotation=45)
        plt.xlabel('Expiry')
        plt.ylabel('Implied Forward Price')
        title = index +': Implied Forward Price vs Expiry'
        title = ''.join(title)
        plt.title(title)
        plt.gca().spines['bottom'].set_color('#a6a6a6')
        plt.gca().spines['top'].set_color('#a6a6a6')
        plt.gca().spines['right'].set_color('#a6a6a6')
        plt.gca().spines['left'].set_color('#a6a6a6')
        plt.gca().tick_params(axis='x', colors='#a6a6a6')
        plt.gca().tick_params(axis='y', colors='#a6a6a6')
        plt.gca().yaxis.label.set_color('#a6a6a6')
        plt.gca().xaxis.label.set_color('#a6a6a6')
        plt.gca().title.set_color('#a6a6a6')
        plt.show()

    # calculate average implied interest rate
    def get_ir(self, implied_params):
        implied_ir_list = implied_params['implied_ir']
        implied_ir_list = implied_ir_list[implied_ir_list > 0]
        implied_ir = implied_ir_list.mean()

        return implied_ir

    def get_iv(self, option_data, implied_params, plot_iv_scatter = False):
        # initialization
        expiry_list = option_data['Expiry'].unique()
        implied_vol = pd.DataFrame(columns=expiry_list)

        for expiry in expiry_list:
            # for each expiry
            r = implied_params[implied_params['Expiry'] == expiry]['implied_ir'].values[0]
            option_price_expiry = self.option_price[self.option_price['Expiry'] == expiry]
            option_strike = option_price_expiry['Strike'].unique()

            for strike in option_strike:
                # for each strike
                option_price_strike = option_price_expiry[option_price_expiry['Strike'] == strike]

                # check if call price is NaN
                if not option_price_strike['c'].isna().values[0]:
                    option_price_strike_k = option_price_strike['c'].values[0]
                    isCall = True
                if not option_price_strike['p'].isna().values[0]:
                    option_price_strike_k = option_price_strike['p'].values[0]
                    isCall = False

                F = implied_params[implied_params['Expiry'] == expiry]['implied_fwd'].values[0]
                K = strike
                T = (pd.to_datetime(expiry, format='%Y-%m-%d') - self.today).days / 365

                implied_vol_expiry_K = bisection_implied_vol(F, K, T, r, option_price_strike_k, isCall=isCall)
                implied_vol.loc[strike, expiry] = implied_vol_expiry_K

        implied_vol = implied_vol.sort_index()
        implied_vol = implied_vol.transpose()
    
        # plot the rought implied volatility as scatter plot
        index = implied_params['Index'].unique()
        if plot_iv_scatter:
            plt.figure(figsize=(10, 6))
            for strike in implied_vol.index:
                plt.scatter(implied_vol.columns, implied_vol.loc[strike])
            plt.xticks(rotation=45)
            plt.xlabel('Strike')
            plt.ylabel('Implied Volatility')
            title = index +': Implied Volatility vs Expiry'
            title = ''.join(title)
            plt.title(title)
            plt.legend(implied_vol.index)
            for text in plt.gca().get_legend().get_texts():
                text.set_color('#a6a6a6')
            plt.gca().spines['bottom'].set_color('#a6a6a6')
            plt.gca().spines['top'].set_color('#a6a6a6')
            plt.gca().spines['right'].set_color('#a6a6a6')
            plt.gca().spines['left'].set_color('#a6a6a6')
            plt.gca().tick_params(axis='x', colors='#a6a6a6')
            plt.gca().tick_params(axis='y', colors='#a6a6a6')
            plt.gca().yaxis.label.set_color('#a6a6a6')
            plt.gca().xaxis.label.set_color('#a6a6a6')
            plt.gca().title.set_color('#a6a6a6')
            plt.gca().get_legend().get_frame().set_alpha(0)
            name = index + '_rough_iv'
            name = ''.join(name)
            if not os.path.exists(f'./Public/Plot/Rough_iv/{name}.png'):
                plt.savefig(f'./Public/Plot/Rough_iv/{name}.png', transparent=True)
            plt.show()
            

        return implied_vol


# ----------------------- test code -----------------------
if __name__ == '__main__':
    from data_cleaning import data_cleaning

    option_data_spx = pd.read_csv('./Public/Data/Option/spx_option_0901.csv')
    option_data_nky = pd.read_csv('./Public/Data/Option/nky_option_0901.csv')
    option_data_hsi = pd.read_csv('./Public/Data/Option/hsi_option_0901.csv')
    
    option_data_spx = data_cleaning(option_data_spx).format_data(index = 'SPX')
    option_data_nky = data_cleaning(option_data_nky).format_data(index = 'NKY')
    option_data_hsi = data_cleaning(option_data_hsi).format_data(index = 'HSI')
    
    option_data_spx = data_cleaning(option_data_spx).check_iv_number(drop_type='volume', drop_threshold=10)
    option_data_nky = data_cleaning(option_data_nky).check_iv_number(drop_type='volume', drop_threshold=2)
    option_data_hsi = data_cleaning(option_data_hsi).check_iv_number()

    spx_data = data_cleaning(option_data_spx).get_hist('SPX', '2021-09-01', '2023-11-13')
    nky_data = data_cleaning(option_data_nky).get_hist('NKY', '2021-09-01', '2023-11-13')
    hsi_data = data_cleaning(option_data_hsi).get_hist('HSI', '2021-09-01', '2023-11-13')
    
    option_price_spx = data_cleaning(option_data_spx).extract_option_price(px_type='mid')
    option_price_nky = data_cleaning(option_data_nky).extract_option_price(px_type='mid')
    option_price_hsi = data_cleaning(option_data_hsi).extract_option_price(px_type='mid')



    option_data = pd.concat([option_data_spx, option_data_nky, option_data_hsi], axis=0)
    option_price = pd.concat([option_price_spx, option_price_nky, option_price_hsi], axis=0)
    
    implied_params = BS_implied_vol(option_price).parity_implied_params(option_data, plot_parity=False)
    implied_vol_spx = BS_implied_vol(option_price[option_price['Index'] == 'SPX']).get_iv(option_data[option_data['Index'] == 'SPX'], implied_params[implied_params['Index'] == 'SPX'], plot_iv_scatter=True)
