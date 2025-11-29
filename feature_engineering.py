def create_features(df):
    df=df.copy()
    df['total_stay_nights']=df['no_of_weekend_nights']+df['no_of_week_nights']
    df['total_guests']=df['no_of_adults']+df['no_of_children']
    df['total_guests']=df['total_guests'].replace(0,1)
    df['avg_price_per_person']=df['avg_price_per_room']/df['total_guests']
    df['weekend_booking_flag']=(df['no_of_weekend_nights']>0).astype(int)
    df['lead_time_category']=df['lead_time'].apply(lambda x:'short' if x<=30 else 'medium' if x<=90 else 'long')
    return df
