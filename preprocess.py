import pandas as pd

def clean_data(df: pd.DataFrame):
    df=df.drop_duplicates()
    df=df.fillna(df.median(numeric_only=True))
    return df

def preprocess_data(df):
    from sklearn.model_selection import train_test_split
    y=df['booking_status'].map({'Canceled':1,'Not_Canceled':0})
    X=df.drop(columns=['booking_status','Booking_ID'])
    return train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
