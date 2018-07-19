#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:06:41 2018

@author: tyler
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm


class PooledOLS():
    def __init__(self, panel, y):
        self.panel = panel
        self.y = str(y)
        
    def fit(self):
        self.y_dep = self.panel[self.y] 
        self.X = self.panel.drop(self.y, axis = 1)
        mod = sm.OLS(self.y_dep, self.X)
        res = mod.fit()
        self.OLScov_ = res.cov_params
        print(res.summary())
        return(res)

class FEEstimator():
    def __init__(self, panel, y):
        self.panel = panel
        self.y = str(y)
        
    def timeinv_mean(self):
        return self.panel.mean(level = 1)
    
    def time_demean(self, panel):
        self.demeaned = self.panel.subtract(self.timeinv_mean(), level = 1) 
        return self.demeaned
    def fit(self):
        demeaned_dataset = self.time_demean(self.panel)
        self.demean_y = demeaned_dataset[self.y] 
        self.demean_X = demeaned_dataset.drop(self.y, axis = 1)
        mod = sm.OLS(self.demean_y, self.demean_X)
        res = mod.fit()
        print(res.summary())
        return(res)
        
        
class BetweenEstimator():
    def __init__(self, panel, y):
        self.panel = panel
        self.y = str(y)
        
    def timeinv_mean(self, panel):
        return self.panel.mean(level = 1)
    
    def fit(self):
        between_dataset = self.timeinv_mean(self.panel)
        self.timemean_y = between_dataset[self.y] 
        self.timemean_X = between_dataset.drop(self.y, axis = 1)
        mod = sm.OLS(self.timemean_y, self.timemean_X)
        res = mod.fit()
        self.residuals_ = self.timemean_y - res.predict()
        print(res.summary())
        return(res)

        

class REEstimator():
    def __init__(self, panel, y):
        self.panel = panel
        self.y = str(y)
        
    def timeinv_mean(self):
        self.means = self.panel.mean(level = 1)
        
    def lambda_create(self):
        self.between = BetweenEstimator(self.panel, self.y)
        self.between.fit()
        self.between_rss = np.sum((self.between.residuals_)**2)
        self.lamb = self.between_rss/(len(self.panel) - 
                          len(self.panel.columns) + 1)

        OLS = PooledOLS(self.panel, self.y)
        self.res = OLS.fit()
        self.OLSvar = np.diag(OLS.OLScov_)
        self.da = self.OLSvar / self.panel.groupby(level=0).size()
        self.lambda_ = self.lamb*self.panel.groupby(level=0).size() - self.da
        if self.lambda_ < 0:
            self.lambda_ == 0
        
        return self.between_rss

    def time_demean(self, panel):
        self.demeaned = self.panel.subtract(self.timeinv_mean(), level = 1) 
        return self.demeaned