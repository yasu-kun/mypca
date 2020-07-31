import sklearn
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd


class pca:

    def __init__(self,n_components=None,normalizing=True):
        self.normalizing = normalizing
        self.n_components = n_components

        #data = data.astype(np.float64)
        
    def normalize(self,data):
        for i in range(data.shape[1]):
            data[:,i] = data[:,i] - np.mean(data[:,i])
        return data

    def _fit(self,X):
        #fitするための準備
        #compornent
        if self.n_components is None:
            self.n_components = X.shape[1]
        else:
            self.n_components = self.n_components
        
        #normalizing 
        #True  => normalizing
        #False => non normalizing
        if self.normalizing is True:
            self.X = self.normalize(X)
        else:
            self.X = X

    def fit(self,X):
        self._fit(X)
        #平均引くか(標準化関係ない)
        self.cov_ver_matrix = np.cov(self.X.T)

        #固有ベクトルは列ベクトル！注意！
        #固有ベクトルは１に規格化されて出力される。
        self.eigen_value, self.eigen_vector = np.linalg.eig(self.cov_ver_matrix)

        #sort
        e_v_sort = np.sort(self.eigen_value)[::-1]
        #累積寄与率
        self.ex_va = np.sum(e_v_sort[:self.n_components]) / np.sum(e_v_sort)

        #内積を計算して主成分得点を計算（v = a1x1 + a2x2）
        p_component = np.array([np.dot(self.eigen_vector.T, self.X[i,:]) for i in range(self.X.shape[0])])
        
        #reshape
        ev = self.eigen_value.reshape(1,self.eigen_value.shape[0])
        #一番上の行に固有値を結合
        p_c = np.concatenate([ev,p_component])
        
        #一番上の行の固有値をもとにソート（降順）
        p_c_s = p_c[:,p_c[0,:].argsort()[::-1]]
        #固有値行を除き、圧縮する次元数を指定。
        self.p_component = p_c_s[1:,:self.n_components]

    '''
    def fit_trance(self):
        #主成分行列
        #a1 = np.array([np.dot(v.T, data[i,:]) for i in range(data.shape[0])])
        p_component = np.array([np.dot(self.eigen_vector.T, self.X[i,:]) for i in range(self.X.shape[0])])
        
        ev = self.eigen_value.reshape(1,self.eigen_value.shape[0])
        p_c = np.concatenate([ev,p_component])
        print(p_c[:,p_c[0,:].argsort()[::-1]])

        #p_component = np.concatenate([self.eigen_value.reshape(1,self.eigen_value.shape[1]),p_component])


        return p_component
    '''
    