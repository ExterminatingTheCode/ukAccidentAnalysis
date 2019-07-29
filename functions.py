import pandas as pd
import numpy as np
from scipy import stats


def load_and_clean_accident_data():
    '''
    This function loads in accident data from /data folder. 

    Source of data: https://www.kaggle.com/daveianhickey/2000-16-traffic-flow-england-scotland-wales
    '''
    
    df1 = pd.read_csv("data/accidents_2005_to_2007.csv")
    df2 = pd.read_csv("data/accidents_2009_to_2011.csv")
    df3 = pd.read_csv("data/accidents_2012_to_2014.csv")

    df12 = pd.concat([df1,df2])
    df = pd.concat([df12,df3])

    #Accident_index column is broken, not all ID's are unique. It appears as though someone's computer converted the string to a number.
    duplicate_cols = ['Accident_Index','Date','LSOA_of_Accident_Location','Time','Longitude','Latitude']

    df = df.drop_duplicates(subset=duplicate_cols, keep='first')


    #Don't see much use for these columns. Removing them makes the data easier to look at.
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



def cleaner_data(df):
    '''
    Removes accidents with either anomalous conditions or missing location data.
    Parameters:
        df: Pandas DataFrame loaded with accident data. 
        
    Returns:
        df: Same Pandas DataFrame but with a few less rows, removing accidents that will not be included. 
    '''
    
    df = df[(df.Road_Surface_Conditions != 'Flood (Over 3cm of water)') & (df.Road_Surface_Conditions !='Snow')]
    df = df[df.Speed_limit !=15]
    df = df[(df.Weather_Conditions == 'Fine without high winds') | (df.Weather_Conditions =='Raining without high winds')]
    df = df.dropna(subset=['Latitude', 'Longitude'])
    return df
    

def find_severity_mus(df):
    '''
    This function finds that, proportion of each severity of accidents compared to all acidents
    '''
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

#Calcualates p values => probability of seeing actual severity given expected severity
def calculate_p_values(actual_severity,expected_severity):
    '''
    Parameters:
        actual_severity: Actual proportion of accidents in a given level of severity in a sample of data
        expected_severity: Hypothesized proportion of accidents in underlying distribution that are in given severity level

    Returns:
        pvalues: probability of seeing data as or more extreme than actual severity given an underlying expected_severity. 
    '''
    n_corr, _ = expected_severity.shape
    pvalues = np.zeros(n_corr)

    for i in np.arange(n_corr):
        pvalues[i] = stats.chisquare(actual_severity[i],expected_severity[i])[1]
    pvalues = pvalues.reshape((-1,1))
    
    return pvalues

def connectTrafficData(accData, trafData, inplace=True):
    ''' 
    Attaches traffic data to accident data as 'Traffic' column
    Parameters:
        accData: Pandas dataframe of the accident data
        trafData: Pandas dataframe of traffic data
        inplace: Default True. If True, will add a "CP" column to accident data with the closest traffic checkpoint. 
            If false will return closest array which can be used to add traffic data. 
    Returns:
        closest: Array of closest traffic CP (checkpoint) and distance to it for each accident in accData. 
    '''
    #Haversine distance finds the actual distance between two points given their latitude and longitude
    #Accuracy for Haversine formula is within 1%, doesn't account for ellipsoidal shape of the earth. 
    from sklearn.metrics.pairwise import haversine_distances

    years = np.unique(accData['Year'])

    # accLocs = accData[['Latitude', 'Longitude']].values
    # trafLocs = trafData[['Lat','Lon']].values

    closest = np.ones((len(accData),3)) * 10
    index = 0

    for year in years:
        curAccs = accData[accData['Year'] == year].copy()
        curTraf = trafData[trafData['year'] == year].copy()
        curAccLocs = curAccs[['Latitude', 'Longitude']].copy().values
        curTrafLocs = curTraf[['latitude', 'longitude']].copy().values
        for i, acc in enumerate(curAccLocs):
            distances = haversine_distances(acc.reshape((1,-1)),curTrafLocs)
            closest[index + i,0] = distances.min()
            CPindex = distances.argmin()
            closest[index + i,1] = curTraf.iloc[CPindex].count_point_id
        index += len(curAccs)
    if inplace:
        accData['CP'] = closest[:,1].copy()
        accData['Traffic'] = closest[:,2].copy()
    else:
        return closest


    def collectTrafficStats(accData, trafData, inplace=True):
        '''
            Parameters:
                accData: Accident Data with traffic checkpoint attached
                trafData: Traffic Data
                inplace: Default True. If true, attaches data to input DataFrame. If False, returns dictionary 

            Returns:
                Dictionary to map new columns if inpace=False, else no return.
        '''

        casualties = accData.groupby(['CP'])['Number_of_Casualties'].agg('sum')
        accidents = accData.groupby(['CP'])['Number_of_Casualties'].agg('count')



                
    #np.save('/Users/mac/galvanize/week4/ukAccidentAnalysis/distance_matrix',closest)