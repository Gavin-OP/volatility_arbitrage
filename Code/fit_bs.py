import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.dates import date2num, num2date
from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
import os
import warnings
from IPython.display import display

# get atm implied vol
def iv_near_atm(x, a, b):
    y = a * x + b
    return y


# define sigmoid function
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


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
    
    def fit_BS_curve(self, fwd_moneyness, bounds = ([0, 2.7, -70], [8, 10, -20]), p0=[4, 5, -30], plot_curve=False, method='tanh'):
        # ignore wanings
        warnings.filterwarnings("ignore")
        # initialization
        expiry_list = self.implied_vol.index.values
        bs_iv_curve_params = pd.DataFrame(columns=['atm_vol', 'delta', 'kappa', 'gamma'])

        for expiry in expiry_list:
            # get fwd mny
            fwd_moneyness_expiry = fwd_moneyness[expiry]
            fwd_moneyness_expiry = fwd_moneyness_expiry.drop(['implied_fwd'])
            fwd_moneyness_expiry = np.log(fwd_moneyness_expiry)

            # get implied vol
            implied_vol_expiry = self.implied_vol.T[expiry]

            # concat fwd moneyness and implied vol, column name: fwd_moneyness, first row: implied_vol
            implied_vol_fwd_moneyness = pd.DataFrame(columns=fwd_moneyness_expiry.values)
            implied_vol_fwd_moneyness.loc['implied_vol'] = implied_vol_expiry.values
            implied_vol_fwd_moneyness = implied_vol_fwd_moneyness.dropna(axis=1)

            # get atm implied vol
            if len(implied_vol_fwd_moneyness.columns) <= 3:
                fwd_moneyness = fwd_moneyness.drop([expiry], axis=1)
                continue
            if 0 not in implied_vol_fwd_moneyness.columns:
                # find 2 columns that closest to 0
                implied_vol_fwd_moneyness.columns = implied_vol_fwd_moneyness.columns.astype(float)
                atm_fit = implied_vol_fwd_moneyness.iloc[:, abs(implied_vol_fwd_moneyness.columns).argsort()[:2]]

                # fit the curve using 2 implied vol near atm 
                fwd_moneyness_near_0_x = atm_fit.columns
                iv_near_0_y = atm_fit.loc['implied_vol'].values
                params, _ = curve_fit(iv_near_atm, fwd_moneyness_near_0_x, iv_near_0_y)
                x = 0
                a = params[0]
                b = params[1]
                atm_vol = iv_near_atm(x, a, b)
            else:
                atm_vol = implied_vol_fwd_moneyness.loc[0]
                
            # fit implied vol curve
            # BS IV curve model
            def implied_vol_curve(x, delta, kappa, gamma):
                if method == 'tanh':
                    y = atm_vol**2 + delta * (np.tanh(kappa * x) / kappa) + 0.5 * gamma * (np.tanh(kappa * x) / kappa)**2
                if method == 'sigmoid':
                    y = atm_vol**2 + delta * (sigmoid(kappa * x) / kappa) + 0.5 * gamma * (sigmoid(kappa * x) / kappa)**2
                return y 

            fwd_moneyness_x = implied_vol_fwd_moneyness.columns.values
            implied_vol_y = implied_vol_fwd_moneyness.loc['implied_vol'].values
            params, _ = curve_fit(implied_vol_curve, fwd_moneyness_x, np.array(implied_vol_y)**2, maxfev=100000)
            
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
            x = np.linspace(-0.15, 0.15, 1000)
            y = np.zeros((len(x), len(bs_iv_curve_params.index)))
            i = 0
            index = self.implied_params['Index'].unique()
            for expiry in bs_iv_curve_params.index:
                # plot fitted curve
                atm_vol = bs_iv_curve_params.loc[expiry]['atm_vol']
                delta = bs_iv_curve_params.loc[expiry]['delta']
                kappa = bs_iv_curve_params.loc[expiry]['kappa']
                gamma = bs_iv_curve_params.loc[expiry]['gamma']
                y[:, i] = implied_vol_curve(x, delta, kappa, gamma)
                y[:, i] = np.sqrt(y[:, i])
                i += 1
                plt.plot(x, y[:, i-1], label=expiry)

                # plot the scattar plot
                fwd_moneyness_expiry = fwd_moneyness[expiry]
                fwd_moneyness_expiry = fwd_moneyness_expiry.drop(['implied_fwd'])
                fwd_moneyness_expiry = np.log(fwd_moneyness_expiry)
                implied_vol_expiry = self.implied_vol.T[expiry]
                plt.scatter(fwd_moneyness_expiry, implied_vol_expiry.values)
            
            title = index + ' BS Implied Volatility Curve'
            title = ''.join(title)
            plt.title(title)
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
            name = index + '_BS_IV_Curve'
            name = ''.join(name)
            if not os.path.exists(f'./Public/Plot/Implied_vol/{name}.png'):
                plt.savefig(f'./Public/Plot/Implied_vol/{name}.png', transparent=True)
            plt.show()
            
        return bs_iv_curve_params
       

    def fit_surface(self, bs_iv_curve_params, fwd_mny = [-0.15, 0.15], step = 20, plot_surface=False, type='wireframe'):
        begin = fwd_mny[0]
        end = fwd_mny[1]
        fwd_mny_grid = np.linspace(begin, end, step)
        implied_vol_expiry = pd.DataFrame(index = fwd_mny_grid, columns = bs_iv_curve_params.index)
        
        # Function to calculate implied volatility using the parametric model
        def implied_vol_curve(x, atm_vol, delta, kappa, gamma):
            y = atm_vol**2 + delta * (np.tanh(kappa * x) / kappa) + 0.5 * gamma * (np.tanh(kappa * x) / kappa)**2
            return y
        
        # Loop over each expiry and calculate implied volatility values
        for i, expiry in enumerate(bs_iv_curve_params.index):
            # Extract parameters for the current expiry
            params = bs_iv_curve_params.loc[expiry].values
            atm_vol = params[0]

            # Calculate implied volatility values for the current expiry
            implied_vol_expiry.loc[:, expiry] = implied_vol_curve(fwd_mny_grid, atm_vol, params[1], params[2], params[3])
        
        # Create a 1D array of unique maturities for CubicSpline
        expiry_list_nu = date2num(bs_iv_curve_params.index)
        expiry_grid = np.linspace(min(expiry_list_nu), max(expiry_list_nu), step)
        
        # Initialize an empty DataFrame to store interpolated implied volatility values
        implied_vol_surface = pd.DataFrame(index=fwd_mny_grid, columns=expiry_grid)
        
        # Perform cubic spline interpolation for each strike
        for i in range(len(fwd_mny_grid)):
            iv_cs = CubicSpline(expiry_list_nu, implied_vol_expiry.iloc[i, :].values)
            implied_vol_surface.iloc[i, :] = iv_cs(expiry_grid)
            
        # Create meshgrid for 3D plotting
        X, Y = np.meshgrid(fwd_mny_grid, expiry_grid)
        
        # Create a 3D plot
        if plot_surface:
            index = self.implied_params['Index'].unique()
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the surface
            if type == 'surface':
                ax.plot_surface(X, Y, implied_vol_surface.T.values, alpha=0.5, edgecolor='#b7d1ff')
            elif type == 'wireframe':
                ax.plot_wireframe(X, Y, implied_vol_surface.T.values, alpha=0.5, edgecolor='#b7d1ff')

            # change the y-axis to date
            expiry_grid = num2date(expiry_grid)
            expiry_grid = [expiry.strftime('%Y-%m-%d') for expiry in expiry_grid]
            ax.set_yticklabels(expiry_grid)

            # Plot the scatter
            ax.scatter(X, Y, implied_vol_surface.T.values, color='#0078d4', alpha=1, s=3)

            # Make the surface transparent
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # Make the grid lines transparent
            ax.xaxis._axinfo["grid"]['color'] =  '#d9d9d9'
            ax.yaxis._axinfo["grid"]['color'] =  '#d9d9d9'
            ax.zaxis._axinfo["grid"]['color'] =  '#d9d9d9'

            # Set labels and title
            title = index + ' Implied Volatility Surface'
            title = ''.join(title)
            ax.set_title(title)
            ax.set_xlabel('Log Forward Moneyness')
            ax.set_ylabel('Term Structure')
            ax.set_zlabel('Volatility')
            
            # make xlabel away from the plot
            ax.yaxis.labelpad = 10

            # Adjust tick size
            ax.tick_params(axis='both', which='major', labelsize=10)
            # set axis color
            ax.xaxis.label.set_color('#c3c3c3')
            ax.yaxis.label.set_color('#c3c3c3')
            ax.zaxis.label.set_color('#c3c3c3')

            # set tick color
            ax.tick_params(axis='x', colors='#c3c3c3')
            ax.tick_params(axis='y', colors='#c3c3c3')
            ax.tick_params(axis='z', colors='#c3c3c3')

            # set axis color
            ax.xaxis.line.set_color('#c3c3c3')
            ax.yaxis.line.set_color('#c3c3c3')
            ax.zaxis.line.set_color('#c3c3c3')

            # Add grid lines with a dashed style
            ax.grid(True, linestyle='--', alpha=0.6)

            ax.view_init(elev=20, azim=135)
            # invert x-axis
            ax.invert_xaxis()
            ax.invert_yaxis()

            # save the plot
            name = index + '_BS_IV_Surface'
            name = ''.join(name)
            if not os.path.exists(f'./Public/Plot/Volatility_surface/{name}.png'):
                plt.savefig(f'./Public/Plot/Volatility_surface/{name}.png', transparent=True, dpi=1000)
            
            plt.show()
            
        return implied_vol_surface



if __name__ == '__main__':
    from data_cleaning import data_cleaning
    from implied_vol import BS_implied_vol

    option_data_spx = pd.read_csv('./Public/Data/Option/spx_option_0901.csv')
    option_data_spx = data_cleaning(option_data_spx).format_data(index = 'SPX')
    option_data_spx = data_cleaning(option_data_spx).check_iv_number(drop_type='volume', drop_threshold=10)
    spx_data = data_cleaning(option_data_spx).get_hist('SPX', '2021-09-01', '2023-11-13')
    option_price_spx = data_cleaning(option_data_spx).extract_option_price(px_type='mid')
    implied_params = BS_implied_vol(option_price).parity_implied_params(option_data, plot_parity=False)
    implied_vol_spx = BS_implied_vol(option_price[option_price['Index'] == 'SPX']).get_iv(option_data[option_data['Index'] == 'SPX'], implied_params[implied_params['Index'] == 'SPX'], plot_iv_scatter=False)
    implied_vol_nky = BS_implied_vol(option_price[option_price['Index'] == 'NKY']).get_iv(option_data[option_data['Index'] == 'NKY'], implied_params[implied_params['Index'] == 'NKY'], plot_iv_scatter=False)
    implied_vol_hsi = BS_implied_vol(option_price[option_price['Index'] == 'HSI']).get_iv(option_data[option_data['Index'] == 'HSI'], implied_params[implied_params['Index'] == 'HSI'], plot_iv_scatter=False)
    
    

    fwd_moneyness_spx = fit_BS(implied_vol_spx, implied_params[implied_params['Index'] == 'SPX']).get_fwd_mny()
    fwd_moneyness_nky = fit_BS(implied_vol_nky, implied_params[implied_params['Index'] == 'NKY']).get_fwd_mny()
    fwd_moneyness_hsi = fit_BS(implied_vol_hsi, implied_params[implied_params['Index'] == 'HSI']).get_fwd_mny()

    bs_iv_curve_params_spx = fit_BS(implied_vol_spx, implied_params[implied_params['Index'] == 'SPX']).fit_BS_curve(fwd_moneyness_spx, plot_curve=True)
    bs_iv_curve_params_nky = fit_BS(implied_vol_nky, implied_params[implied_params['Index'] == 'NKY']).fit_BS_curve(fwd_moneyness_nky, plot_curve=True, method='tanh')
    bs_iv_curve_params_hsi = fit_BS(implied_vol_hsi, implied_params[implied_params['Index'] == 'HSI']).fit_BS_curve(fwd_moneyness_hsi, plot_curve=True)

    bs_iv_surface_spx = fit_BS(implied_vol_spx, implied_params[implied_params['Index'] == 'SPX']).fit_surface(bs_iv_curve_params_spx, plot_surface=True, type='wireframe')
    bs_iv_surface_nky = fit_BS(implied_vol_nky, implied_params[implied_params['Index'] == 'NKY']).fit_surface(bs_iv_curve_params_nky, plot_surface=True)
    bs_iv_surface_hsi = fit_BS(implied_vol_hsi, implied_params[implied_params['Index'] == 'HSI']).fit_surface(bs_iv_curve_params_hsi, plot_surface=True)
