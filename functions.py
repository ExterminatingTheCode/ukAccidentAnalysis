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

def load_new_accident_data():
    '''
    This function loads in accident data from /data folder. 

    Source of data: https://www.kaggle.com/silicon99/dft-accident-data
        Source of that source: https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data
    '''

    df = pd.read_csv("data/Accidents0515.csv")

    df.dropna(subset=['Latitude'],inplace=True)

    Year = np.zeros(len(df))
    for i, dateString in enumerate(df['Date']):
        Year[i] = int(dateString[-4:])
    df['Year'] = Year

    return df

def load_and_clean_traffic_data():
    '''
    This function loads in accident data from /data folder

    source : https://roadtraffic.dft.gov.uk/downloads

    '''
    # Directly read in the data
    traffic = pd.read_csv("data/TrafficData/TotalTraffic.csv")

    # Do not consider checkpoints that have "0" traffic
    traffic = traffic[traffic['all_motor_vehicles'] > 0].copy()

    #Do not consider checkpoints that have zero length of road (results in division by zero)
    traffic = traffic[traffic['link_length_km'] > 0].copy()

    #Do not consider checkpoints with nan for link length, same as above
    traffic.dropna(subset=['link_length_km'],inplace=True)

    return traffic




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

def connectTrafficData(accData, trafData, inplace=True, hardsave=False):
    ''' 
    Attaches traffic data to accident data as 'Traffic' column
    Parameters:
        accData: Pandas dataframe of the accident data
        trafData: Pandas dataframe of traffic data
        inplace: Default True. If True, will add a "CP" column to accident data with the closest traffic checkpoint. 
            If false will return closest array which can be used to add traffic data. 
        hardsave: Default False. If true will save the resulting DataFrame in the Data directory. 

    Returns:
        closest: Array of closest traffic CP (checkpoint) and distance to it for each accident in accData. 
    '''
    #Haversine distance finds the actual distance between two points given their latitude and longitude
    #Accuracy for Haversine formula is within 1%, doesn't account for ellipsoidal shape of the earth. 
    from sklearn.metrics.pairwise import haversine_distances

    years = np.unique(accData['Year'])

    # accLocs = accData[['Latitude', 'Longitude']].values
    # trafLocs = trafData[['Lat','Lon']].values

    closest = np.ones((len(accData),5)) * 10
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
            closest[index + i,2] = curTraf.iloc[CPindex].all_motor_vehicles
            closest[index + i,3] = curTraf.iloc[CPindex].latitude
            closest[index + i,4] = curTraf.iloc[CPindex].longitude
        index += len(curAccs)
    if inplace:
        accData['CP'] = closest[:,1].copy()
        accData['Traffic'] = closest[:,2].copy()
        accData['CPlatitude'] = closest[:,3].copy()
        accData['CPlongitude'] = closest[:,4].copy()
        if hardsave:
            accData.to_csv("data/accidents_2005_to_2014_wTraffic.csv")
    else:
        return closest
                        
    #np.save('/Users/mac/galvanize/week4/ukAccidentAnalysis/distance_matrix',closest)


def collectTrafficStats(accData, trafData, inplace=True, hardsave = False):
    '''
        Parameters:
            accData: Accident Data with traffic checkpoint attached
            trafData: Traffic Data
            inplace: Default True. If true, attaches data to input DataFrame. If False, returns traffic  
            hardsave: Default False, if both inplace and hardave are true than the traffic dataframe is hardsaved

        Returns:
            Only returns if inplace is False
            Casualties: numpy array of the number of casualties for each Checkpoint and each year
            , Accidents: numpy array of the number of accidents at each point for each year in the dataset 
    '''

    casualties = accData.groupby(['CP','Year'])['Number_of_Casualties'].agg('sum')
    accidents = accData.groupby(['CP','Year'])['Number_of_Casualties'].agg('count')

    if inplace:
        trafData['num_casualties'] = casualties
        trafData['num_accidents'] = accidents
        if hardsave:
            trafData.to_csv('data/TrafficStatistics.csv')
    else:
        return casualties, accidents

def mainGraphAccidentData(accData, CoastLatitudes=None, CoastLongitudes=None):
    '''
    Paramters:

    Returns:

    '''
    import matplotlib.pyplot as plt

    df1 = accData[accData['Year'] == 2005].copy()
    df1 = df1[df1['Location_Northing_OSGR'] < 990000].copy()
    df1 = df1[df1['Longitude'] > -6.3].copy()

    latitudes = (df1.Latitude - df1.Latitude.values.min())/(df1.Latitude.values.max() - df1.Latitude.values.min())
    longitudes = (df1.Longitude.values - df1.Longitude.values.min())/(df1.Longitude.values.max() - df1.Longitude.values.min())

    fig, ax = plt.subplots(figsize= (6,9.5))

    if CoastLatitudes:
        ax.plot(CoastLatitudes, CoastLongitudes, alpha=1, c='black')

    ax.scatter(longitudes, latitudes, s=1, alpha=1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.set_facecolor('xkcd:white')
    None


def graphSubsetAccidentData(accData,latitude=None ,longitude =None, CoastLatitudes=None, CoastLongitudes=None):
    '''
    Paramters:

    Returns:

    '''
    import matplotlib.pyplot as plt

    df1 = accData[accData['Year'] == 2005].copy()
    if latitude:
        df1 = df1[df1['Latitude'] > latitude].copy()
        df1 = df1[df1['Longitude'] > longitude].copy()
        CoastLatitudes = CoastLatitudes[CoastLatitudes < Latitude]
        CoastLongitudes = CoastLongitudes[CoastLongitudes < Longitude]
    else:
        df1 = df1[df1['Location_Northing_OSGR'] < 990000].copy()
        df1 = df1[df1['Longitude'] > -6.3].copy()

    latitudes = (df1.Latitude - df1.Latitude.values.min())/(df1.Latitude.values.max() - df1.Latitude.values.min())
    longitudes = (df1.Longitude.values - df1.Longitude.values.min())/(df1.Longitude.values.max() - df1.Longitude.values.min())

    fig, ax = plt.subplots(figsize= (6,9.5))

    if CoastLatitudes:
        ax.plot(CoastLatitudes, CoastLongitudes, alpha=1, c='black')

    ax.scatter(longitudes, latitudes, s=1, alpha=1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.set_facecolor('xkcd:white')
    None