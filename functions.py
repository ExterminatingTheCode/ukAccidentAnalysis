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


