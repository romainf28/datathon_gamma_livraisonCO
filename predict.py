import xgboost as xgb
import pandas as pd
from preprocessing import get_weather_forecast,add_holidays
from datetime import datetime

def create_features(df, label=None):
    df['date'] = df["Date et heure de comptage"]
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour', 'dayofweek','month', 'quarter', 'year', 'dayofyear',
            'dayofmonth', 'weekofyear','Jour férié', 'Vacances scolaires','Durée avant les prochaines vacances scolaires','tempC','visibility','cloudcover','precipMM','humidity']]
    if label:
        y = df[label]
        return X, y
    return X

def make_predictions(df_road,df_test,arc_name,datetimes):
    all_preds = []
    for metric in ['Débit horaire',"Taux d'occupation"]:
        X_train,y_train = create_features(df_road,label=metric)
        X_test = create_features(df_test)

        X_train['Durée avant les prochaines vacances scolaires'] = pd.to_timedelta(X_train['Durée avant les prochaines vacances scolaires']).dt.days
        X_test['Durée avant les prochaines vacances scolaires'] = pd.to_timedelta(X_test['Durée avant les prochaines vacances scolaires']).dt.days



        reg = xgb.XGBRegressor(n_estimators=1000)
        reg.fit(X_train, y_train,
                verbose = False)

        preds = reg.predict(X_test)
        all_preds.append(preds)
    return pd.DataFrame({"arc":[arc_name]*len(datetimes),"datetime":datetimes,"debit_horaire":all_preds[0], "taux_occupation":all_preds[1]})
            


def main():
    DATA_DIR = "./data"
    df_ce = pd.read_csv('dataframes/df_champs_elysees.csv')
    df_sts = pd.read_csv('dataframes/df_saints_peres.csv')
    df_conv = pd.read_csv('dataframes/df_saints_peres.csv')

    df_ce['Date'] = pd.to_datetime(df_ce['Date'])
    df_sts['Date'] = pd.to_datetime(df_sts['Date'])
    df_conv['Date'] = pd.to_datetime(df_conv['Date'])

    df_ce['Date et heure de comptage'] = pd.to_datetime(df_ce['Date et heure de comptage'])
    df_sts['Date et heure de comptage'] = pd.to_datetime(df_sts['Date et heure de comptage'])
    df_conv['Date et heure de comptage'] = pd.to_datetime(df_conv['Date et heure de comptage'])

    df_ce = df_ce[df_ce['Date']<datetime(2022,12,9)]
    df_sts = df_sts[df_sts['Date']<datetime(2022,12,9)]
    df_conv = df_conv[df_conv['Date']<datetime(2022,12,9)]

    df_weather_forecast = get_weather_forecast(DATA_DIR)
    df_weather_forecast['Date et heure de comptage'] = df_weather_forecast['date_time']
    df_weather_forecast['Date'] = pd.to_datetime(df_weather_forecast["date_time"]).dt.date

    df = df_weather_forecast

    add_holidays(df,DATA_DIR)

    df = df[(df['Date et heure de comptage'] < datetime(2022,12,14)) & (df['Date et heure de comptage'] >= datetime(2022,12,9))]

    datetimes = df['Date et heure de comptage']
    df_preds_ce = make_predictions(df_ce,df,'Champs Elysées',datetimes)
    df_preds_conv = make_predictions(df_conv,df,'Convention',datetimes)
    df_preds_sts = make_predictions(df_sts,df,'Saints-Pères',datetimes)

    df_all_preds = pd.concat([df_preds_ce,df_preds_conv,df_preds_sts],axis=0)
    df_all_preds.to_csv('predictions/output_epsilon_consulting.csv')

if __name__ == "__main__":
    main()
        
    

