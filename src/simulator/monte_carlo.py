import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


class MonteCarloSimulator:
    def __init__(self):
        pass
    
        
    def dist_model_norm(self, df, t_ahead):
        days = (df['adj_close'].index[-1] - df['adj_close'].index[0]).days
        mu = ((((df['adj_close'][-1]) / df['adj_close'][0])) ** (365.0/days)) - 1
        vol = df['daily_returns'].std()*np.sqrt(t_ahead)
        
        #This might not be right TODO
        return np.random.normal((mu/252), vol/np.sqrt(252), t_ahead)+1
    
    def dist_model_brownian(self, df, t_ahead): 
        
        log_returns = np.log(1 + df['adj_close'].pct_change())
        u = log_returns.mean()
        var = log_returns.var()
        drift = u - (0.5 * var)
        stdev = log_returns.std()

        daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_ahead)))
        return daily_returns

    
    
    def get_distmodel(self, df, t_ahead, model = "norm"):
        """Returns a sampeling method for daily stock returns
        """
        if(model == "norm"):
            return self.dist_model_norm(df, t_ahead)
        elif(model == "linear"):
            return self.dist_model_norm(df, t_ahead)
        elif(model == "brownian"):
            return self.dist_model_brownian(df, t_ahead)
        
    def simulat_steps(self, df_in, T = None, t_ahead = 252, model = "norm", nb_sim = 1000, plot = True):
        """
        T = look back, how much many days of old data do we simulate upon
        t_ahead = days ahead simulated
        df_in = dataframe with date indexing and cols ["adj_close", "daily_returns"]
        """
        #list of results
        result = []
        
        if(T == None):
            df = df_in
        else:
            df = df_in.tail(T)
        
        for i in range(nb_sim):
            
            #costum daily returns list
            daily_returns = self.get_distmodel(df, t_ahead, model)
            
            #current simulation price_list
            start = df['adj_close'].values[-1]
            price_list = [start]
            
            
            for x in daily_returns:
                price_list.append(price_list[-1]*x)

            result.append(price_list[-1])

            if(plot):
                plt.plot(price_list, color='b', alpha= np.min([1.0,20./nb_sim]))


        mean = np.mean(result)
        tile_5 = np.percentile(result,5)
        tile95 = np.percentile(result,95)
        if(plot):
            plt.show()

            plt.hist(result,bins=100)
            plt.axvline(np.percentile(result,5), color='r', linestyle='dashed', linewidth=2)
            plt.axvline(np.percentile(result,95), color='r', linestyle='dashed', linewidth=2)
            plt.show()


        return {"start" : start,
                "mean" : np.mean(result),
                "5%" : np.percentile(result,5),
                "95%" : np.percentile(result,95)}
        
