from wwo_hist import retrieve_hist_data
import os
os.chdir("../data")

frequency=1
start_date = '01-NOV-2021'
end_date = '09-DEC-2022'
api_key = 'fb8420267edf4ae0be9130732220412'
location_list = ['Paris']

hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)