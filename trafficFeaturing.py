import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd


def load_data():
    traffic = pd.read_csv("data/ukTrafficAADF.csv")
    traffic = traffic[traffic['AllMotorVehicles'] != 0]

    return traffic

def clean_data(data):
    cols_to_drop = ['Region', 'LocalAuthority', 'Road',
                    'Easting','Northing', 'StartJunction', 'EndJunction', 'LinkLength_km']
    return data.drop(cols_to_drop,axis=1)

def add_previous(traffic, columns_to_copy):
    '''This function will append the previous years' value of columns [2:29]
        in columns [29: ]
    Parameters:
    
    traffic: Pandas dataframe including traffic data

    '''

    # columns_to_copy = traffic.columns[2:]

    feature_engineering = traffic.copy()

    len_previous = feature_engineering.shape[1]

    for col in columns_to_copy:
        feature_engineering[('previous_' + col)] = traffic[col]

    feature_engineering.sort_values(['CP','AADFYear'],inplace=True)

    columns = feature_engineering.columns
    traf_vals = feature_engineering.values

    for i in range(len(feature_engineering)-1):
        if (traf_vals[i,1] == traf_vals[i+1,1]):
            traf_vals[i+1,len_previous:] = traf_vals[i,3:3+len(columns_to_copy)]

    return pd.DataFrame(traf_vals, columns=columns)


def subset_data(data,columns,conditions):
    pass

def unique_vals(column):
    npcol = np.array(column)

    uniques = np.unique(npcol)

    return uniques
    

def train_test_splitter(data, y):

    n = len(data)
    train = np.zeros(n)
    test = np.zeros(n)

    train = data['AADFYear'] < y
    current = data['AADFYear'] == y
    randoms = current * stats.bernoulli(.8).rvs(n)

    train += randoms
    test = current - randoms

    return train, test

