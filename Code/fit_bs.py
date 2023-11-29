import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import os
import warnings

# get atm implied vol
def iv_near_atm(x, a, b):
    y = a * x + b
    return y

   

class fit_BS():
    def __init__(self, implied_vol, implied_params):
        self.implied_vol = implied_vol
        self.implied_params = implied_params
        
    
    def get_fwd_mny(self):
        # initialization, column name: expiry, first row: implied_fwd
        fwd_moneyness = pd.DataFrame(columns=self.implied_params['Expiry'].unique())
        fwd_moneyness.loc['implied_fwd'] = self.implied_params['implied_fwd'].values

        # get all strike values
        strike_list = self.implied_vol.columns.values.astype(int)

        # for each strike, calculate the forward moneyness
        for strike in strike_list:
            fwd_moneyness.loc[strike] = strike/fwd_moneyness.loc['implied_fwd']
        
        return fwd_moneyness
    
    def fit_BS_curve(self, fwd_moneyness, bounds = ([0, 2.7, -70], [8, 10, -20]), p0=[4, 5, -30], plot_curve=False):
        # ignore wanings
        warnings.filterwarnings("ignore")
        # initialization
        expiry_list = self.implied_vol.index.values
        bs_iv_curve_params = pd.DataFrame(columns=['atm_vol', 'delta', 'kappa', 'gamma'])

        for expiry in expiry_list:
            # get fwd mny
            fwd_moneyness_expiry = fwd_moneyness[expiry]
            fwd_moneyness_expiry = fwd_moneyness_expiry.drop(['implied_fwd'])

            # get implied vol
            implied_vol_expiry = self.implied_vol.T[expiry]

            # concat fwd moneyness and implied vol, column name: fwd_moneyness, first row: implied_vol
            implied_vol_fwd_moneyness = pd.DataFrame(columns=fwd_moneyness_expiry.values)
            implied_vol_fwd_moneyness.loc['implied_vol'] = implied_vol_expiry.values
            implied_vol_fwd_moneyness = implied_vol_fwd_moneyness.dropna(axis=1)

            # get atm implied vol
            if 1 not in implied_vol_fwd_moneyness.columns:
                # find 2 columns that closest to atm
                implied_vol_fwd_moneyness.columns = implied_vol_fwd_moneyness.columns.astype(float)
                atm_fit = implied_vol_fwd_moneyness.iloc[:, abs(implied_vol_fwd_moneyness.columns - 1).argsort()[:2]]

                # fit the curve using 2 implied vol near atm 
                fwd_moneyness_near_1_x = atm_fit.columns
                iv_near_1_y = atm_fit.loc['implied_vol'].values
                params, _ = curve_fit(iv_near_atm, fwd_moneyness_near_1_x, iv_near_1_y)
                x = 1
                a = params[0]
                b = params[1]
                atm_vol = iv_near_atm(x, a, b)
            else:
                atm_vol = implied_vol_fwd_moneyness.loc[1]
                
            # fit implied vol curve
            # BS IV curve model
            def implied_vol_curve(x, delta, kappa, gamma):
                    y = atm_vol**2 + delta * (np.tanh(kappa * x) / kappa) + 0.5 * gamma * (np.tanh(kappa * x) / kappa)**2
                    y = np.sqrt(y)
                    return y

            fwd_moneyness_x = implied_vol_fwd_moneyness.columns.values
            implied_vol_y = implied_vol_fwd_moneyness.loc['implied_vol'].values
            params, _ = curve_fit(implied_vol_curve, fwd_moneyness_x, implied_vol_y, maxfev=1000000, bounds=bounds, p0=p0)
            delta = params[0]
            kappa = params[1]
            gamma = params[2]
            
            # store bs iv curve params
            bs_iv_curve_params = pd.concat([bs_iv_curve_params, pd.DataFrame([[atm_vol, delta, kappa, gamma]], columns=['atm_vol', 'delta', 'kappa', 'gamma'])], axis=0)
        
        # set index
        bs_iv_curve_params.index = fwd_moneyness.columns

        # plot the implied vol curve with scatter plot
        if plot_curve:
            plt.figure(figsize=(10, 6))
            x = np.linspace(0.8, 1.15, 100)
            y = np.zeros((len(x), len(bs_iv_curve_params.index)))
            i = 0
            for expiry in bs_iv_curve_params.index:
                # plot fitted curve
                atm_vol = bs_iv_curve_params.loc[expiry]['atm_vol']
                delta = bs_iv_curve_params.loc[expiry]['delta']
                kappa = bs_iv_curve_params.loc[expiry]['kappa']
                gamma = bs_iv_curve_params.loc[expiry]['gamma']
                y[:, i] = implied_vol_curve(x, delta, kappa, gamma)
                i += 1
                plt.plot(x, y[:, i-1], label=expiry)

                # plot the scattar plot
                fwd_moneyness_expiry = fwd_moneyness[expiry]
                fwd_moneyness_expiry = fwd_moneyness_expiry.drop(['implied_fwd'])
                implied_vol_expiry = self.implied_vol.T[expiry]
                plt.scatter(fwd_moneyness_expiry, implied_vol_expiry.values)
            
            plt.title('BS Implied Volatility Curve')
            plt.xlabel('Forward Moneyness')
            plt.ylabel('BS Implied Volatility')
            for text in plt.legend().get_texts():
                text.set_color("#a6a6a6")
            plt.gca().spines['bottom'].set_color('#a6a6a6')
            plt.gca().spines['top'].set_color('#a6a6a6')
            plt.gca().spines['right'].set_color('#a6a6a6')
            plt.gca().spines['left'].set_color('#a6a6a6')
            plt.gca().tick_params(axis='x', colors='#a6a6a6')
            plt.gca().tick_params(axis='y', colors='#a6a6a6')
            plt.gca().yaxis.label.set_color('#a6a6a6')
            plt.gca().xaxis.label.set_color('#a6a6a6')
            plt.gca().title.set_color('#a6a6a6')
            plt.legend()
            plt.gca().get_legend().get_frame().set_alpha(0)
            name = 'BS_IV_Curve'
            if not os.path.exists(f'./Public/Plot/Implied_vol/{name}.png'):
                plt.savefig(f'./Public/Plot/Implied_vol/{name}.png', transparent=True)
            plt.show()
            
        return bs_iv_curve_params
       


if __name__ == '__main__':
    fit_BS_curve(p0=[1, 7, -65])