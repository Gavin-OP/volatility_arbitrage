# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from functools import partial


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
    # define BS model
    def BS_fwd_pricer(F, K, T, sig, r, isCall):
        d1 = (np.log(F / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        
        if isCall:
            option_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            option_price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

        return option_price
    
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
        expiry_list = option_data['Expiry'].unique()
        implied_params = pd.DataFrame(columns=['Expiry', 'implied_ir', 'implied_fwd'])
        # today = pd.to_datetime('2023-09-01', format='%Y-%m-%d')

        for expiry in expiry_list:
            # for each expiry
            option_price_expiry = self.option_price[self.option_price['Expiry'] == expiry]
            option_price_expiry = option_price_expiry.dropna()
            option_price_expiry['c-p'] = option_price_expiry['c'] - option_price_expiry['p']

            # regress c-p on -strike
            model = LinearRegression()
            x = option_price_expiry[['Strike']]
            y = option_price_expiry['c-p']
            T = (pd.to_datetime(expiry, format='%Y-%m-%d') - self.today).days / 365
            model.fit(x, y)
            discount_factor = -model.coef_[0]
            
            # get implied risk free rate and implied forward price
            forward_price = model.intercept_ / discount_factor
            risk_free_rate = -np.log(discount_factor) / T

            # format the output
            implied_params = pd.concat([implied_params, pd.DataFrame({'Expiry': expiry, 'implied_ir': risk_free_rate, 'implied_fwd': forward_price}, index=[0])], axis=0)
            
            # plot c-p vs strike
            if plot_parity:
                plt.figure(figsize=(10, 6))
                plt.scatter(option_price_expiry['Strike'], option_price_expiry['c-p'])
                plt.plot(option_price_expiry['Strike'], model.predict(x), color='red')
                plt.text(0.6, 0.8, f'c-p = {model.coef_[0]:.4f} * strike + {model.intercept_:.4f}', transform=plt.gca().transAxes)
                plt.xlabel('Strike')
                plt.ylabel('Call - Put')
                title = 'Call - Put vs Strike for ' + expiry
                plt.title(title)
                plt.show()
        
        # reset index
        implied_params = implied_params.reset_index(drop=True)

        return implied_params

    # plot implied ir vs expiry
    def plot_ir(self, implied_params):
        plt.figure(figsize=(10, 6))
        plt.plot(implied_params['Expiry'], implied_params['implied_ir'])
        plt.ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.xlabel('Expiry')
        plt.ylabel('Implied Interest Rate')
        plt.title('Implied Interest Rate vs Expiry')
        plt.show()
    
    # plot implied fwd vs expiry
    def plot_fwd(self, implied_params):
        plt.figure(figsize=(10, 6))
        plt.plot(implied_params['Expiry'], implied_params['implied_fwd'])
        plt.xticks(rotation=45)
        plt.xlabel('Expiry')
        plt.ylabel('Implied Forward Price')
        plt.title('Implied Forward Price vs Expiry')
        plt.show()

    # calculate average implied interest rate
    def get_ir(self, implied_params):
        implied_ir_list = implied_params['implied_ir']
        implied_ir_list = implied_ir_list[implied_ir_list > 0]
        implied_ir = implied_ir_list.mean()

        return implied_ir

    def get_iv(self, option_data, implied_params, spx_data, r, plot_iv_scatter = False):
        # initialization
        expiry_list = option_data['Expiry'].unique()
        implied_vol = pd.DataFrame(columns=expiry_list)

        for expiry in expiry_list:
            # for each expiry
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
        
        # calculate forward moneyness
        spx_0901 = spx_data.loc['2023-09-01', 'Open']
        implied_vol.columns = implied_vol.columns.astype(float) / spx_0901
    
        # plot the rought implied volatility as scatter plot
        if plot_iv_scatter:
            plt.figure(figsize=(10, 6))
            for strike in implied_vol.index:
                plt.scatter(implied_vol.columns, implied_vol.loc[strike])
            plt.xticks(rotation=45)
            plt.xlabel('Strike')
            plt.ylabel('Implied Volatility')
            plt.title('Implied Volatility vs Expiry')
            plt.legend(implied_vol.index)
            plt.show()
            

        return implied_vol


# ----------------------- test code -----------------------
if __name__ == '__main__':
    print('1')
