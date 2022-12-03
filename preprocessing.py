import pandas as pd
import os 
from functools import reduce

def read_csv_files(data_dir,filenames):
    dfs = [pd.read_csv(os.path.join(data_dir, f), sep=';', index_col=0).assign(filename=f) for f in filenames]

    for idx, df in enumerate(dfs):
        df.set_index("Date et heure de comptage",drop=True,inplace=True)
        df = df.sort_index()
        df['Débit horaire'] = df['Débit horaire'].interpolate()
        df["Taux d'occupation"] = df["Taux d'occupation"].interpolate()
        
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
    df[date_col] = pd.to_datetime(df.index,format='%Y-%m-%d %H:%M',utc=True)
    df[date_col] = df[date_col].apply(lambda d : d.tz_localize(None) + pd.DateOffset(hours=1))
    return df

def is_during_holiday(date,df_holidays):
    for _, row in df_holidays.iterrows():
        start = row['start_date']
        end = row['end_date']
        
        if start.date() <= date <= end.date():
            return True
    return False


def add_holidays(df,data_dir):
    df_bank_holiday = pd.read_csv(os.path.join(data_dir,'jours_feries_metropole.csv'))
    df_bank_holiday = df_bank_holiday[df_bank_holiday['annee'].isin([2021, 2022])]
    df['Jour férié'] = df['Date'].isin( pd.to_datetime(df_bank_holiday['date']).apply(lambda d : d.date()))

    df_holiday = pd.read_csv(os.path.join(data_dir,'fr-en-calendrier-scolaire.csv'), sep=';')
    df_holiday = df_holiday[df_holiday['location'] == 'Paris']
    df_holiday = df_holiday[df_holiday['annee_scolaire'].isin(['2021-2022', '2022-2023'])]
    df_holiday['start_date'] = pd.to_datetime(df_holiday['start_date'])
    df_holiday['end_date'] = pd.to_datetime(df_holiday['end_date'])
    df['Vacances scolaires'] = df['Date'].apply(lambda d : is_during_holiday(d, df_holiday) )


