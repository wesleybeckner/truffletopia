import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
import joblib

def load_model():
    model = joblib.load("data/customer_forcast.joblib")
    return model

def process_data(df):
    """
    ### ASSUMPTIONS ###
    only 1 date column and 1 target (floating point) value column
    the rest are groupby columns
    """
    
    # infer dates
    date_col = None
    groupby = []
    target_col = None
    for col in df.columns:
        if df[col].dtype == np.object:
            try:
                df[col] = pd.to_datetime(df[col])
                date_col = col
            except:
                groupby.append(col)
        else:
            target_col = col

    # reset index to prep for pivot
    if len(groupby) > 0:
        df = df.set_index(groupby)

    # expand out date for regression/model fitting
    # value will automatically be the target_col
    df = df.pivot(columns=date_col)

    # remove extraneous units label (result form th epivot)
    df.columns = df.columns.droplevel()
    df = df.reset_index()
    
    return df

def train_test_split(df, groupby=None, train_frac=0.8, verbiose=False):
    """
    Transform "tidy data" s.t. datetime features are expanded column-wise. Split
    into train/test sets across datetime-unparsed instances so that true testing
    examples are intact.
    
    Notes
    -----
    Datetime columns now contain what were previously field values in the df object
    so they must be inferred.
    
    groupby must be passed as list
    
    Parameters
    ----------
    df: dataframe
    gorupby: list
    train_frac: float
    verbiose: bool
    
    Returns
    -------
    X_train: array
    X_test: array
    qty: list
    enc: OneHotEncoder
    """
    
    # qty will now be a piece of meta data we need to keep for window_sweep
    # becuase it is all of the datetime columns
    qty = list(df.select_dtypes(exclude=datetime).columns)
      
    if groupby:
        # create the encoder object
        enc = OneHotEncoder()
        # grab the columns we want to convert from strings
        X_cat = df[groupby].values.reshape(-1,len(groupby))
        # fit our encoder to this data
        enc.fit(X_cat)
        onehotlabels = enc.transform(X_cat).toarray()
        X = np.concatenate((onehotlabels, df[qty].values), axis=1)
    else:
        X = df[qty].values
        enc = None
        
    np.random.seed(4)
    np.random.shuffle(X)
    X_train = X[:int(X.shape[0]*train_frac),:]
    X_test = X[int(X.shape[0]*train_frac):,:]
    
    if verbiose:
        print(X_train.shape, X_test.shape)
        
    return X_train, X_test, qty, enc

def sweep_window(X_train, qty, window=3, verbiose=False):
    """
    Assumes the non-datetime columns are first, and then the
    datetime columns, qty specifies where this split happens
    """
    X_cat = X_train[:,:-len(qty)]
    X = X_train[:,-len(qty):]
    X_ = []
    y = []
    for i in range(X.shape[1]-window):
        X_.append(np.concatenate((X_cat, X[:, i:i+window]), axis=1))
        y.append(X[:, i+window])
    X_ = np.array(X_).reshape(X.shape[0]*np.array(X_).shape[0],window+X_cat.shape[1])
    y = np.array(y).reshape(X.shape[0]*np.array(y).shape[0],)
    X_ = X_[np.where(~np.isnan(y))[0]]
    y = y[np.where(~np.isnan(y))[0]]

    labels = []
    for row in X_:
        labels.append("X: {}".format(np.array2string(row[-3:].round())))
    
    if verbiose:
        print(X_.shape, y.shape)
    return X_, y, labels

def make_forcast(model, df, window, qty, delta='month', time_delta_previous=12, groupby='customer', item='Perk-a-Cola', projection=18):
    """
    There are parameters that are specific for train and some for production. 
    groupby/item will allow filteration of the dataset
    to only return values according to those filters.
    
    Parameters
    ----------
    model: sklearn model
        default RandomForestRegressor under models/
    df: DataFrame
        the data for the model
    window: int (training param)
        the window used for sweep_window call
    qty: list (training param)
        the datetime column names, used to create new projected timesteps and
        to autodetermine the end timestep of the dataset
    delta: string (production param, default month)
        year, month, week, day, used to create new projected timesteps
    time_delta_previous: int (default 12)
        previous timesteps to inclue alongside predictions
    groupby: string, bool (default customer)
        column to perform filter
    item: string, (default Perk-a-Cola)
        instance to filter along filter/groupby column
    projection: int (defaul 18)
        timesteps to project into future
    
    Returns
    -------
    dff: DataFrame
        the projected/prediction dataframe
    """
    
    if groupby != None:
        df = df.loc[df[groupby] == item].copy()
    else:
        pass

    # make newtimes to run forecast on
    delta = 'month'
    year = qty[-1].year
    month = qty[-1].month
    day = qty[-1].day

    
    newtimes = []
    for i in range(1,projection+1):
        if delta == 'month':
            month += 1
            if month == 13:
                month = 1
                year += 1
        else:
            print('delta not set')
            break
        time = datetime(year, month, day)
        newtimes.append(time)
        
    dff = pd.DataFrame()
    
    # when y-target is in training data
    for i in range(time_delta_previous):
        month = qty[-(time_delta_previous-i)]
        X = df[qty[-(window + time_delta_previous - i) : -(time_delta_previous-i)]].values
        y = df[month].values
        month = [month]*X.shape[0]
        pred = model.predict(X)
        dff = pd.concat([dff, 
                         pd.DataFrame([pred,y,month]).T])

    # when X is in training data
    month = [newtimes[0]]*X.shape[0]
    X = df[qty[-window:]].values
    pred = predgrow = model.predict(X)
    dff = pd.concat([dff,
                    pd.DataFrame([pred,[None]*X.shape[0],month]).T])

    # when X is both in training data and prior time step predictions
    for i in range(1,projection):
        j = projection - i
        X = df[qty[-(window-i):]].values
        X = np.concatenate((X, predgrow.reshape(-1,i)), axis=1)
        month = [newtimes[i]]*X.shape[0]
        pred = model.predict(X)
        predgrow = np.hstack((predgrow, pred))
        dff = pd.concat([dff,
                        pd.DataFrame([pred,[None]*X.shape[0],month]).T])

    dff = dff.reset_index(drop=True)
    dff.columns =['Prediction', 'Actual', 'Month']
    dff = dff.melt(id_vars='Month', var_name='Source', value_name='KG')
    
    return dff