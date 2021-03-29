# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:53:04 2021

@author: 4440
"""
import numpy as np
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import math
import random
import gensim
from gensim import corpora

class PLSA(object):
    def __init__(self, N, Z):
        self.N = N
        self.X = N.shape[0]
        self.Y = N.shape[1]
        self.Z = Z

        # P(z)
        self.Pz = np.random.rand(self.Z)
        # P(x|z)
        self.Px_z = np.random.rand(self.Z, self.X)
        # P(y|z)
        self.Py_z = np.random.rand(self.Z, self.Y)

        # 正規化
        self.Pz /= np.sum(self.Pz)
        self.Px_z /= np.sum(self.Px_z, axis=1)[:, None]
        self.Py_z /= np.sum(self.Py_z, axis=1)[:, None]

        self.beta = 1
        self.eta = 0.9999

    def train(self, k=2000, t=1.0e-7):
        '''
        対数尤度が収束するまでEステップとMステップを繰り返す
        '''
        prev_llh = 100000
        for i in range(k):
            self.e_step()
            self.m_step()
            llh = self.llh()

            print('llh',llh)

            if abs((llh - prev_llh) / prev_llh) < t:
                break

            prev_llh = llh

    def e_step(self):
        '''
        Eステップ
        P(z|x,y)の更新
        '''
        # normal EM
        # self.Pz_xy = self.Pz[None, None, :] * self.Px_z.T[:, None, :] * self.Py_z.T[None, :, :]

        # Tempered EM
        self.Pz_xy = self.Pz[None, None, :] * ((self.Px_z.T[:, None, :] * self.Py_z.T[None, :, :])**self.beta)


        self.Pz_xy /= np.sum(self.Pz_xy, axis=2)[:, :, None]

        self.Pz_xy[np.isnan(self.Pz_xy)] = 0
        self.Pz_xy[np.isinf(self.Pz_xy)] = 0
        self.beta = self.beta * self.eta

        # print('beta',self.beta)

    def m_step(self):
        '''
        Mステップ
        P(z), P(x|z), P(y|z)の更新
        '''
        NP = self.N[:, :, None] * self.Pz_xy

        self.Pz = np.sum(NP, axis=(0, 1))
        self.Px_z = np.sum(NP, axis=1).T
        self.Py_z = np.sum(NP, axis=0).T

        self.Pz /= np.sum(self.Pz)
        self.Px_z /= np.sum(self.Px_z, axis=1)[:, None]
        self.Py_z /= np.sum(self.Py_z, axis=1)[:, None]

    def llh(self):
        '''
        対数尤度
        '''
        Pxy = self.Pz[None, None, :] * self.Px_z.T[:, None, :] * self.Py_z.T[None, :, :]
        Pxy = np.sum(Pxy, axis=2)
        Pxy /= np.sum(Pxy)
        Pxy[Pxy < 0] = 0
        self.N[self.N < 0] = 0
        loglh = np.sum(self.N * np.log1p(Pxy))
        return loglh
    
if __name__ == "__main__":
    noun_path = '申告内容_plsa_8.csv'
    noun_df = pd.read_csv(noun_path, encoding='UTF-8', header=0)
    dependency_path = '問合理由_plsa_8.csv'
    dependency_df = pd.read_csv(dependency_path, encoding='UTF-8', header=0)
    Co_occurrence_matrix = pd.DataFrame(index=noun_df.columns[4:-2], columns=dependency_df.columns[4:-2]).fillna(0)

"""     
    #可読性ZERO地獄4重ループ
    for dependency_index, dependency_row in tqdm(dependency_df.iterrows()):
        for noun_index, noun_row in noun_df.iterrows():
            if dependency_row['行ID']-1 == noun_row['行ID']:
                for dependency_key, dependency_value in dependency_row[4:-2].iteritems():
                    for noun_key, noun_value in noun_row[4:-2].iteritems():
                        if noun_value + dependency_value == 2:
                            Co_occurrence_matrix.at[noun_key, dependency_key] += 1
"""     
    plsa = PLSA(Co_occurrence_matrix.values, 10)
    plsa.train()

    # print ('P(z)')
    # print (plsa.Pz)
    # print ('P(x|z)')
    # print (plsa.Px_z.shape)
    # print ('P(y|z)')
    # print (plsa.Py_z)

    topics = ['word', 'topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6', 'topic7', 'topic8', 'topic9', 'topic10']
    x_matrix = pd.DataFrame(index=[], columns=topics)
    y_matrix = pd.DataFrame(index=[], columns=topics)
    
    # print ('P(z|x)')
    Pz_x = plsa.Px_z.T * plsa.Pz[None, :]
    # Pz_x = Pz_x / np.sum(Pz_x, axis=1)[:, None]
    
    for i, x in enumerate(Co_occurrence_matrix.index):
        # x_row = pd.Series([x, Pz_x[i][0], Pz_x[i][1], Pz_x[i][2], Pz_x[i][3], Pz_x[i][4], Pz_x[i][5], Pz_x[i][6], Pz_x[i][7], Pz_x[i][8], Pz_x[i][9]], index=x_matrix.columns)
        x_row = pd.Series([x, plsa.Px_z.T[i][0], plsa.Px_z.T[i][1], plsa.Px_z.T[i][2], plsa.Px_z.T[i][3], plsa.Px_z.T[i][4], plsa.Px_z.T[i][5], plsa.Px_z.T[i][6], plsa.Px_z.T[i][7], plsa.Px_z.T[i][8], plsa.Px_z.T[i][9]], index=x_matrix.columns)
        x_matrix = x_matrix.append(x_row, ignore_index=True)
    
    # print ('P(z|y)')
    Pz_y = plsa.Py_z.T * plsa.Pz[None, :]
    # Pz_y = Pz_y / np.sum(Pz_y, axis=1)[:, None]
    for i, y in enumerate(Co_occurrence_matrix.columns):
        # y_row = pd.Series([y, Pz_y[i][0], Pz_y[i][1], Pz_y[i][2], Pz_y[i][3], Pz_y[i][4], Pz_y[i][5], Pz_y[i][6], Pz_y[i][7], Pz_y[i][8], Pz_y[i][9]], index=y_matrix.columns)
        y_row = pd.Series([y, plsa.Py_z.T[i][0], plsa.Py_z.T[i][1], plsa.Py_z.T[i][2], plsa.Py_z.T[i][3], plsa.Py_z.T[i][4], plsa.Py_z.T[i][5], plsa.Py_z.T[i][6], plsa.Py_z.T[i][7], plsa.Py_z.T[i][8], plsa.Py_z.T[i][9]], index=y_matrix.columns)
        y_matrix = y_matrix.append(y_row, ignore_index=True)
    
    # x_matrix.to_csv('申告内容_plsa_result_pxz.csv')
    # y_matrix.to_csv('問合理由_plsa_result_pyz.csv')
        
    
    Pdx = 1/noun_df.iloc[:,4:-2]
    Pdy = 1/dependency_df.iloc[:,4:-2]
    Pdx[np.isinf(Pdx)] = 0
    Pdy[np.isinf(Pdy)] = 0
    Pdxz = np.dot(Pdx.values, plsa.Px_z.T)
    Pdyz = np.dot(Pdy.values, plsa.Py_z.T)
    Pddx = 0.5
    Pddy = 0.5
    Pdz = (Pddx * Pdxz) + (Pddy * Pdyz)
    Pd = np.dot(Pdz, plsa.Pz)
    score = Pdz.T/Pd
    score[np.isnan(score)] = 0
    df = pd.DataFrame(score.T)
    df['原文'] = noun_df['原文']
    # df.to_csv('result.csv')
    
    score.shape
    # Pdyz.shape