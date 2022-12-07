import pandas as pd
import os 
from functools import reduce

def read_csv_files(data_dir,filenames):
    dfs = [pd.read_csv(os.path.join(data_dir, f), sep=';', index_col=0).assign(filename=f) for f in filenames]

    for idx, df in enumerate(dfs):
        df= convert_dates_to_datetime(df,'Date et heure de comptage')
        df = df.set_index("Date et heure de comptage",drop=False)
        df.index.names = ['ds']
        df= df.sort_index()
        dfs[idx] = df
    
    return dfs

def filter_nodes(df):
    noeuds_amont_aval = {
        'champs-elysees.csv': {
            'amont':'Av_Champs_Elysees-Washington',
            'aval':'Av_Champs_Elysees-Berri'
        },
        'convention.csv': {
            'amont':'Lecourbe-Convention',
            'aval': 'Convention-Blomet'
        },
        'saints-peres.csv': {
            'amont':'Sts_Peres-Voltaire',
            'aval': 'Sts_Peres-Universite'
        }
    }
    filtering_criteria = []
    for k, v in noeuds_amont_aval.items():
        criterion = (df['filename']==k) & (df['Libelle noeud amont']==v['amont']) & (df['Libelle noeud aval']==v['aval'])
        filtering_criteria.append(criterion)

    mask = reduce(lambda x, y: x | y, filtering_criteria)
    return df[mask]

def convert_dates_to_datetime(df,date_col):
    df[date_col] = pd.to_datetime(df[date_col],format='%Y-%m-%d %H:%M',utc=True)
    df[date_col] = df[date_col].apply(lambda d : d.tz_localize(None) + pd.DateOffset(hours=1))
    return df

def fill_missing_values(df):
    df['Débit horaire'] = df.apply(lambda x: df[df['Date et heure de comptage'] == (x['Date et heure de comptage'] - pd.offsets.DateOffset(weeks=1))]['Débit horaire'] if pd.isna(x['Débit horaire']) else x['Débit horaire'], axis=1)
    df["Taux d'occupation"] = df.apply(lambda x: df[df['Date et heure de comptage'] == x['Date et heure de comptage']-  pd.offsets.DateOffset(weeks=1)]["Taux d'occupation"] if pd.isna(x["Taux d'occupation"]) else x["Taux d'occupation"], axis=1)

def is_during_holiday(date,df_holidays):
    for _, row in df_holidays.iterrows():
        start = row['start_date']
        end = row['end_date']
        
        if start.date() <= date <= end.date():
            return True
    return False

def first_holiday_after(date,sorted_df_holidays):
    for _,row in sorted_df_holidays.iterrows():
        start = row['start_date']
        if start.date() > date:
            return start.date()
    raise Exception(f'No holiday after {date}')

def add_holidays(df,data_dir):
    df_bank_holiday = pd.read_csv(os.path.join(data_dir,'jours_feries_metropole.csv'))
    df_bank_holiday = df_bank_holiday[df_bank_holiday['annee'].isin([2021, 2022])]
    df['Jour férié'] = df['Date'].isin( pd.to_datetime(df_bank_holiday['date']).apply(lambda d : d.date()))

    df_holiday = pd.read_csv(os.path.join(data_dir,'fr-en-calendrier-scolaire.csv'), sep=';')
    df_holiday = df_holiday[df_holiday['location'] == 'Paris']
    df_holiday = df_holiday[df_holiday['annee_scolaire'].isin(['2021-2022', '2022-2023'])]
    df_holiday['start_date'] = pd.to_datetime(df_holiday['start_date'])
    df_holiday['end_date'] = pd.to_datetime(df_holiday['end_date'])

    df_holiday = df_holiday.sort_values(by=['start_date'])
    df['Vacances scolaires'] = df['Date'].apply(lambda d : is_during_holiday(d, df_holiday) )
    df['Prochaines vacances scolaires'] = pd.to_datetime(df['Date'].apply(lambda d : first_holiday_after(d,df_holiday)))
    df['Durée avant les prochaines vacances scolaires'] = df['Prochaines vacances scolaires'] - df['Date et heure de comptage']

def add_weather_data(df,data_dir):
        df_weather = pd.read_csv(os.path.join(data_dir,'weather_paris.csv'))
        df_weather['date_time'] = pd.to_datetime(df_weather['date_time'])
        return df.merge(df_weather, left_on = 'Date et heure de comptage', right_on='date_time', how='left').drop(columns=['date_time'])

def get_train_test(df):
    cut_date = df['Date et heure de comptage'].max()-pd.DateOffset(days=5)
    df_train = df[df['Date et heure de comptage']<cut_date]
    df_test = df[df['Date et heure de comptage']>= cut_date]
    return df_train, df_test

def set_indexes_for_timeseries(df):
    df = df.set_index('Date et heure de comptage')
    df = df.asfreq('H', method= 'ffill')
    df.sort_values('Date et heure de comptage',inplace=True)
    return df
