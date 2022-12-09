import requests
import json
import csv

params = {'q':'Paris,France','num_of_days':10,'key':'fb8420267edf4ae0be9130732220412','format':'csv','tp':1}
response = requests.get("http://api.worldweatheronline.com/premium/v1/weather.ashx",params=params)
with open('../data/weather_forecast_paris.csv', 'w') as f:
    writer = csv.writer(f)
    for line in response.iter_lines():
        writer.writerow(line.decode('utf-8').split(','))