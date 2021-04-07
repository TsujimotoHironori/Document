# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:22:27 2021

@author: 4440
"""
# ライブラリーインポート、グラフ描画準備
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
#from statsmodels.tsa import stattools as st
from statsmodels.tsa.stattools import adfuller
import datetime
import os

# Dickey-Fuller test 結果と標準偏差、平均のプロット
def test_stationarity(timeseries, window_size=12):
    # Determing rolling statistics
    #当日を含めた過去12ヶ月の平均と標準偏差
    rolmean = timeseries.rolling(window=window_size,center=False).mean()
    rolstd = timeseries.rolling(window=window_size,center=False).std()

    # Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value',
                                             '#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#インプットデータの変換用
def my_date_parser(dt):
    return datetime.datetime.strptime(dt, '%Y%m')

plt.style.use('ggplot')

if __name__ == '__main__':
    # PandasでCSVファイルを読み込む
    filepass = 'UM_20140401_20200831_月単位.csv'
    df1 = pd.read_csv(filepass,parse_dates=[0],date_parser=my_date_parser,encoding='Shift_JIS',index_col=[0])
    df = df1.rename(columns={'販売単価_再計算_平均':'#Passengers'})
    ts = df['#Passengers']
    print("データの長さ",len(df)) 
    save_dir = r'C:\\Users\\4440\\Box\\★ＡＩラボラトリ\\01_ユーザー案件\\05_住友林業\\合板需要予測\\継続データ分析支援\\作業内容\\3月作業\\モデルの構築\\結果\\'

    
    #y軸のオフセット表現を無くし、データをグラフで可視化(pandas.plot)
    plt.gca().ticklabel_format(style='plain', axis='y')
    ts.plot(figsize=(8, 2))
    
    # 自己相関(acf)のグラフ、ラグ1は1日前との相関係数、ラグ2は2日前との相関係数
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts, lags=36, ax=ax1)
    
    # 偏自己相関(pacf)のグラフ、自己相関のグラフでは2日前との相関を見るのに時系列や1日前のデータが関係してきてしまっている？
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=36, ax=ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acf_pacf.png')) 

    # オリジナル ->トレンド成分、季節成分、残差成分に分解してプロット
    res = sm.tsa.seasonal_decompose(ts.values, period=20)
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid
    
    plt.figure(figsize=(8, 8))
    
    # オリジナルの時系列データプロット
    plt.subplot(411)
    plt.plot(ts)
    plt.ylabel('Original')
    
    # trend のプロット、トレンド成分を見ることで長期の変化を見る
    plt.subplot(412)
    plt.plot(trend)
    plt.ylabel('Trend')
    
    # seasonal のプロット
    plt.subplot(413)
    plt.plot(seasonal)
    plt.ylabel('Seasonality')
    
    # residual のプロット
    plt.subplot(414)
    plt.plot(residual)
    plt.ylabel('Residuals')
    plt.savefig(os.path.join(save_dir, 'trend.png')) 

    plt.tight_layout()
    plt.show(block=False)
       
    #単位根かを検定　⇒　単位根の場合は共和分過程化を確認
    #単位根であり共和分過程でない場合は時系列モデル、共和分過程である場合は回帰モデルが推奨？
    
    #対象データが単位根かを検定。帰無仮説：単位根である。対立仮説：単位根ではない　＞　p値0.07
    test_stationarity(ts, window_size=12)
    
    #共和分過程の確認
    
    #単位根であったため、差分を取って定常過程なるかを確認　＞　p値0.007
    diff = ts - ts.shift()
    diff = diff.dropna()
    test_stationarity(diff, window_size=12)
