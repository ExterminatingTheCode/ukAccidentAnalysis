import pandas as pd
import numpy as np
from scipy import stats


def load_and_clean_accident_data():
    df1 = pd.read_csv("data/accidents_2005_to_2007.csv")
    df2 = pd.read_csv("data/accidents_2009_to_2011.csv")
    df3 = pd.read_csv("data/accidents_2012_to_2014.csv")

    df12 = pd.concat([df1,df2])
    df = pd.concat([df12,df3])

    duplicate_cols = ['Accident_Index','Date','LSOA_of_Accident_Location','Time','Longitude','Latitude']

    df = df.drop_duplicates(subset=duplicate_cols, keep='first')

    unnecessary_columns = (['Police_Force'
        ,'Local_Authority_(Highway)'
        ,'Local_Authority_(District)'
        ,'Pedestrian_Crossing-Human_Control'
        ,'Pedestrian_Crossing-Physical_Facilities'
        ,'Did_Police_Officer_Attend_Scene_of_Accident']
    )

    for col in unnecessary_columns:
        del df[col]

    return df

def find_severity_mus(df):
    severity_totals = df.groupby('Accident_Severity').size().values
    #total is 1,469,963 number of accidents 1469963
    severity_mus = severity_totals / severity_totals.sum()
    return severity_mus


# This function is no longer used because I realized dot product is better
def calculate_expected_severities(severity_mus,corr_sums):
    n_mus = len(severity_mus)

    n_cats = len(corr_sums)
    print(n_mus, " ",n_cats)
    expected = np.zeros((n_cats,n_mus))
    for i in np.arange(n_mus):
        for j in np.arange(n_cats):
            expected[i][j] = severity_mus[j] * corr_sums[i]
    expected = expected.round().astype(int)
    return expected

def calculate_p_values(actual_severity,expected_severity):
    n_corr, _ = expected_severity.shape
    pvalues = np.zeros(n_corr)

    for i in np.arange(n_corr):
        pvalues[i] = stats.chisquare(actual_severity[i],expected_severity[i])[1]
    pvalues = pvalues.reshape((-1,1))
    
    return pvalues
