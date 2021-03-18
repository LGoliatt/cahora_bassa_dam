#!/usr/bin/python
# -*- coding: utf-8 -*
import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")


from sklearn.decomposition import PCA, KernelPCA
from scipy.spatial import distance
from sklearn.cluster import * #DBSCAN,KMeans,MeanShift,Ward,AffinityPropagation,SpectralClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import kneighbors_graph
#from sklearn.mixture import GMM, DPGMM
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from xlrd import open_workbook,cellname,XL_CELL_TEXT
#import openpyxl as px
import scipy as sp
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import  TimeSeriesSplit

#from mvpa2.suite import SimpleSOMMapper # http://www.pymvpa.org/examples/som.html
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import euclidean_distances

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

import numpy as np
import pylab as pl
import matplotlib.pyplot as pl
import pandas as pd
import os
import sys
import re
from scipy import stats
#from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import glob
import seaborn as sns
#from sklearn.cross_validation import cross_val_score, ShuffleSplit, LeaveOneOut, LeavePOut, KFold
#-------------------------------------------------------------------------------
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[np.int(window_len/2-1):-np.int(window_len/2)]

def split_sequences_multivariate_days_ahead(sequences, n_steps_in, n_steps_out):
    # split a multivariate sequence into samples
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix+1 > len(sequences):
			break
		# gather input and output parts of the pattern
		#seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix+1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)    
#------------------------------------------------------------------------------- 
#%% 
def read_data_cahora_bassa_sequence(
            filename='./data/data_cahora_bassa/cahora-bassa.csv',
            look_back=21,  look_forward=7, unit='day', kind='ml',
            roll=False, window=5, scale=True, ecological=True,
        ):
    #%%
    #look_back=7; look_forward=7; kind='ml'; unit='day'; roll=True; window=7; scale=False
    filename='./data/data_cahora_bassa/cahora-bassa.csv'
    df= pd.read_csv(filename,  delimiter=';')
    df.index = pd.DatetimeIndex(data=df['Data'].values, dayfirst=True)
    df['year']=[a.year for a in df.index]
    df['month']=[a.month for a in df.index]
    df.drop('Data', axis=1, inplace=True) 
    #df['week']=[a.week for a in df.index]
    #
    if unit=='day':
        df['day']=[a.day for a in df.index]
        #df = df.groupby(['day', 'month', 'year']).agg(np.mean)
        #dt = pd.DataFrame([ {'year':int(y), 'month':int(m), 'day':d} for (d,m,y) in df.index.values] )        
        dt=df.index
        df.index=pd.to_datetime(dt, yearfirst=True)
    elif unit=='month':   
        df_std = df.groupby(['month', 'year']).agg(np.std)
        df = df.groupby(['month', 'year']).agg(np.mean)
        dt = pd.DataFrame([ {'year':int(y), 'month':int(m), 'day':15} for (m,y) in df.index.values] )
        df.index=pd.to_datetime(dt, yearfirst=True)
    else:
        sys.exit('Time slot is not defined: day or month')
        
    #>>df['Time']= [a.year+a.dayofyear/366 for a in df.index]   
    df.sort_index(inplace=True)
    #
    idx = df.index < '2013-12-31'
    idx = df.index < '2015-12-31'
    idx = df.index < '2018-12-31'
    df=df[idx]
    
    c='Caudal Afluente (m3/s)'
    df[c]/=1e3
    out_seq=df[c]
    #aux=df.rolling(window=5, min_periods=1, win_type=None).sum()
    #df['Prec. Acum. (mm)']=aux['Precipitacao (mm)']
        
    #df['smooth']=smooth(df[c].values, window_len=10)
    #if unit=='day':
    #    df[c]=smooth(df[c].values, window_len=10)
        
    clstdrp=[]#['Precipitacao (mm)', 'Evaporacao (mm)','Humidade Relativa (%)',]
    if unit=='day':
        cols_to_drop = clstdrp+['Cota (m)', 'Caudal Efluente (m3/s)', 'Volume Evaporado (mm3)',  'year', 'month', 'day']
    elif unit=='month':   
        cols_to_drop = clstdrp+['Cota (m)', 'Caudal Efluente (m3/s)', 'Volume Evaporado (mm3)', ]
    else:
        sys.exit('Time slot is not defined: day or month')
    
    df.drop(cols_to_drop, axis=1, inplace=True) 
    
    #df.drop(df.columns, axis=1, inplace=True); df[c]=out_seq
    df.columns=['Q', 'R', 'E', 'H', ]

    pl.rc('text', usetex=True)
    pl.rc('font', family='serif',  serif='Times')
    df=df[df.index<'2015-12-31']
    pl.figure(figsize=(10,15))
    for i,group in enumerate(df.columns):
        pl.subplot(len(df.columns), 1, i+1)
        df[group].plot(marker='.',fontsize=16,)#pyplot.plot(dataset[group].values)
        pl.title(group, y=0.5, loc='right')
        pl.axvline('2012-06-30', color='k')
    pl.show()

        
    feature_names=df.columns    
    df['Target']=out_seq 
    target_names=[c]
    dates = df.index

    # if unit=='month':
    #     pl.plot(df.index, df[target_names].values)    
    #     pl.fill_between(df.index, 
    #                     (df[target_names].values - df_std[target_names].values).ravel(), 
    #                     (df[target_names].values + df_std[target_names].values).ravel(), 
    #              alpha=0.2, color='k')

    if scale:       
        scaler=preprocessing.MinMaxScaler()
        scaled=scaler.fit_transform(df)
        scaled=pd.DataFrame(data=scaled, columns=df.columns, index=df.index)
    else:
        scaled=df
    
    #ds = df.values
    ds = scaled.values
    n_steps_in, n_steps_out = look_back, look_forward
    X, y = split_sequences_multivariate_days_ahead(ds, n_steps_in, n_steps_out)
    y = y[:,-1].reshape(-1,1); n_steps_out = y.shape[1] # inly the last day
    dates=df.index[look_forward+look_back-1::]
    
    if roll:
         y_roll=pd.DataFrame(y).rolling(window=window, min_periods=1, win_type=None).mean().values
         y=y_roll
    
    if ecological:
         y_eco=pd.DataFrame(y).rolling(window=7, min_periods=1, win_type=None).min().values
         y=y_eco
    
    train_set_date = '2014-12-31' 
    train_set_date = '2012-06-30' 
    train_size, test_size = sum(dates <= train_set_date), sum(dates > train_set_date) 
    X_train, X_test = X[0:train_size], X[train_size:len(dates)]
    y_train, y_test = y[0:train_size], y[train_size:len(dates)]
    #y_std_train, y_std_test = df_std[target].values[0:train_size], df_std[target].values[train_size:len(dates)]
        
    pl.figure(figsize=(16,4)); 
    pl.plot([a for a in y_train]+[None for a in y_test]);
    pl.plot([None for a in y_train]+[a for a in y_test]); 
    pl.show()

    mnth=[a.month for a in dates]
    n_samples, _, n_features = X_train.shape
    if kind=='ml':        
        X_train = np.array([list(X_train[i].T.ravel()) for i in range(len(X_train))])
        X_test  = np.array([list(X_test[i].T.ravel()) for i in range(len(X_test))])
        y_train, y_test = y_train.T, y_test.T 
        n_features = n_features+look_back
        #X_train=np.c_[X_train, mnth[:train_size]]
        #X_test=np.c_[X_test, mnth[train_size:]]
        n_samples, n_features = X_train.shape
        feature_names=np.array([ str(i)+'_{-'+str(j)+'}' for i in feature_names for j in range(look_back)])
    
    data_description = np.array(['var_'+str(i) for i in range(n_features)])
    dataset=  {
      'task'            : 'regression',
      'name'            : 'Cahora Bassa '+str(look_back)+' '+unit+'s back '+str(look_forward)+' '+unit+'s ahead',
      'feature_names'   : feature_names,
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train,
      'X_test'          : X_test,
      'y_train'         : y_train,
      'y_test'          : y_test,      
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : data_description,
      'items'           : None,
      'reference'       : "Alfeu",      
      'normalize'       : 'MinMax',
      'date_range'      : {'train': dates[0:train_size], 'test': dates[train_size:]},
      }
    #%%
    return dataset
#%%
#%%
#
#
#
if __name__ == "__main__":
    datasets = [
                 #read_data_energy_appliances(),
                 #read_data_hotspot(day=1, plot=True),
                 #read_data_hotspot(day=2, plot=True),
                 #read_data_hotspot(day=3, plot=True),
                 #read_data_british_columbia_daily_sequence(look_back=10, look_forward=1, kind='ml', roll=False,),
                 #read_data_host_guest(),
                 #read_data_british_columbia_daily_sequence(look_back=1, look_forward=1, kind='ml', roll=True, window=5,),
                 #read_data_bituminous_4var(),
                 #read_data_tran2019(),
                 #read_data_ldc_tayfur(),
                 #read_data_efficiency(),
#                read_data_burkina_faso_boromo(),
#                read_data_burkina_faso_dori(),
#                read_data_burkina_faso_gaoua(),
#                read_data_burkina_faso_po(),
#                read_data_burkina_faso_bobo_dioulasso(),
#                read_data_burkina_faso_bur_dedougou(),
#                read_data_burkina_faso_fada_ngourma(),
#                read_data_burkina_faso_ouahigouy(),
#                read_data_b2w(),
#                read_data_qsar_aquatic(),
                 #read_data_cahora_bassa(),
                 #read_data_iraq_monthly(),
                read_data_cahora_bassa_sequence(look_back=14, look_forward=1, kind='ml', unit='day'  , roll=True, window=7),                
                #read_data_cahora_bassa_sequence(look_back=12, look_forward=3, kind='ml', unit='month', roll=False,),
                 #read_data_cergy(),
#                read_data_bogas(),
#                read_data_dutos_csv(),
#                read_data_yeh(),
#                read_data_lim(),
#                read_data_siddique(),
#                read_data_pala(),
#                read_data_bituminous_marshall(),
#                read_data_slump(),
#                read_data_borgomano(),
#                read_data_xie_dgf(),
#                read_data_xie_hgf(),
#                read_data_nguyen_01(),
#                read_data_nguyen_02(),
#                read_data_tahiri(),
#                read_data_alameer_sequence(),
                #read_data_escl(),
                #read_data_beyca2019(),
                #read_data_shamiri(),
                #read_data_copper_workpiece(material='iron'),
                #read_data_copper_workpiece(material='copper'),
                #read_data_turning(experiment='cryogenic'),
                #read_data_turning(experiment='dry'),
            ]
    for D in datasets:
        print('='*80+'\n'+D['name']+'\n'+'='*80)
        print(D['reference'])
        print( D['y_train'])
        print('\n')
#%%-----------------------------------------------------------------------------
