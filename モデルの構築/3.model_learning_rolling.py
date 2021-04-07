# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:23:52 2021

@author: 4440
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA,ARMA
import datetime
import numpy as np
import os
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore') # 計算警告を非表示
from monthdelta import monthmod

#インプットデータの変換用
def my_date_parser(dt):
    return datetime.datetime.strptime(dt, '%Y%m')

#構築したモデルの残差を確認し、周期性を確認
def resid_savefig(resid,name):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=30, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(resid, lags=20, ax=ax2)
    plt.show()
    plt.savefig(os.path.join(save_dir,"resid_" + name + '.png'))
    return

#予測値を結合
def predict_concat(output,model_predict,name):
    output_model = pd.DataFrame(model_predict)
    output_model.columns = [name+"の予測値"]
    output = pd.concat([output,output_model],axis=1, join='outer')
    return output

plt.style.use('ggplot')

if __name__ == '__main__':
    # PandasでCSVファイルを読み込む
    filepass = 'UM_20140401_20200831_月単位.csv'
    df1 = pd.read_csv(filepass,parse_dates=[0],date_parser=my_date_parser,encoding='Shift_JIS',index_col=[0])
    df = df1.rename(columns={'販売単価_再計算_平均':'y'})#列名を変更する
    save_dir = r'C:\\Users\\4440\\Box\\★ＡＩラボラトリ\\01_ユーザー案件\\05_住友林業\\合板需要予測\\継続データ分析支援\\作業内容\\3月作業\\モデルの構築\\結果\\'

    #単位根のため、差分を取って定常過程にする
    diff = df['y'] - df['y'].shift()
    diff = diff.dropna()
    train_parameter = diff.loc['2014-04-01':'2018-01-01']

    #検証と学習に分割するために目的変数を抽出
    ts = df['y']

    #train, testデータに分割
    train = ts.loc['2014-04-01':'2018-01-01']
    test = ts.loc['2018-01-01':'2020-08-01']
    
    train.plot(figsize=(8, 2))
    test.plot(figsize=(8, 2))
    plt.ylabel('Original',fontname="MS Gothic")
    plt.show(block=False)

    #aicを基準にARMAモデルの次数を決める
    res = sm.tsa.arma_order_select_ic(train_parameter, ic='aic', trend = 'c')
    print("ARMAモデルの最適次数",res)
    
    #予測範囲を設定
    pre_start="2014-05-01"
    pre_end="2021-08-01"
    data_start="2014-04-01"    
    data_end="2020-08-01"
    data_end_next = (datetime.datetime.strptime(data_end, '%Y-%m-%d')) + relativedelta(months=1)
    
    #ARIMAモデルの構築、p,d,q
    arima  = ARIMA(train,order=[1,1,1]).fit(dist=False,disp=0) 
    print(arima.summary())    
    arima_predict = arima.predict(start=pre_start,end=pre_end)

    #arimaモデルの返り値は、差分(Yt+1 - Yt)のため、Yt+1 = Yt + 予測値へ変換
    ts.loc[data_end_next] = 0.00
    arima_predict_v = ts.loc[data_start:].shift(1) + arima_predict[pre_start:]

    #実測値が存在しない月をARIMAで予測。
    non_data = monthmod(datetime.datetime.strptime(data_end, '%Y-%m-%d'),datetime.datetime.strptime(pre_end, '%Y-%m-%d'))
    for i in range(non_data[0].months - 1):
        after = datetime.datetime.strptime(data_end,'%Y-%m-%d') + relativedelta(months=i+2)
        before = datetime.datetime.strptime(data_end,'%Y-%m-%d') + relativedelta(months=i+1)
        arima_predict_v[after] = arima_predict_v[before] + arima_predict[after]
    
    #実測値が0のままなので、nullに戻す
    ts.loc[data_end_next] = np.nan

    #SARIMAモデルの構築、p,d,q,P,D,Q,s
    sarima  = sm.tsa.SARIMAX(train,trend="c",order=[1,1,1],
                             seasonal_order=(1,1,1,24),
                             enforce_stationarity=False, 
                             enforce_invertibility=False
                             ).fit(ic='aic',disp=0)
    print(sarima.summary())
    sarima_predict = sarima.predict(pre_start,pre_end)
    
    #y軸のオフセット表現を無くす
    plt.gca().ticklabel_format(style='plain', axis='y')
    
    #実測値と予測値のプロット,赤色：実測値、青色：1番目にplt,紫色：2番目にplt,濃い緑：3番目にplt
    ts.plot(figsize=(8, 2))
    arma_predict.plot(figsize=(8, 2))
    arima_predict_v.plot(figsize=(8, 2))
    sarima_predict.plot(figsize=(8, 2))
    plt.ylabel('Original',fontname="MS Gothic")
    plt.show()
    plt.savefig(os.path.join(save_dir, 'predict.png')) 
    
    #各モデル 残差のチェック、周期性の確認
    resid_savefig(arma.resid,"arma")
    resid_savefig(arima.resid,"arima")
    resid_savefig(sarima.resid,"sarima")
    
    #csvファイルへ出力する準備
    output = pd.DataFrame(ts)
    output.rename(columns={"y":"販売単価の実測値"},inplace=True)
    
    #予測値を結合
    output = predict_concat(output,arma_predict,"arma")
    output = predict_concat(output,arima_predict_v,"arima")
    output = predict_concat(output,sarima_predict,"sarima")

    #絶対値誤差の作成    
    output["arma_絶対値誤差"] = np.abs(output["販売単価の実測値"] - output["armaの予測値"])
    output["arima_絶対値誤差"] = np.abs(output["販売単価の実測値"] - output["arimaの予測値"])
    output["sarima_絶対値誤差"] = np.abs(output["販売単価の実測値"] - output["sarimaの予測値"])

    #出力
    output["年月"] = output.index
    output["年月"] = pd.DatetimeIndex(output["年月"])
    output['データ区分'] = np.where(output['年月'] >= datetime.datetime(2018,8,1),"test",'train')
    output.to_csv("sammury.csv", encoding='utf_8_sig')