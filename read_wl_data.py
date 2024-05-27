import pandas as pd
import os
import numpy as np
import json

def standard_timestamp(time):
    std_time = time.split(sep=':')[0] + ":00:00"
    return std_time

def extract_sensor_data(sensor_data):
    data_per_hour = {}
    dict_data = {'timestamp':[standard_timestamp(time) for time in sensor_data["result_timestamp"].to_list()],"avg_water_level":sensor_data["water_level_filtered_avg_water_level_validated"].to_list(),"min_water_level":sensor_data["water_level_filtered_min_water_level_validated"].to_list(),"max_water_level":sensor_data["water_level_filtered_max_water_level_validated"].to_list()}
    for ind_t in range(len(dict_data['timestamp'])):
        if not np.isnan(dict_data['avg_water_level'][ind_t]):
            time = dict_data['timestamp'][ind_t]
            avg_wl = dict_data['avg_water_level'][ind_t]
            max_wl = dict_data['max_water_level'][ind_t]
            min_wl = dict_data['min_water_level'][ind_t]
            data_per_hour[time] = {'avg':avg_wl,'max':max_wl,'min':min_wl}
    dict_data['data_per_hour'] = data_per_hour
    return dict_data

def same_hour(time1,time2):
    day_hour1 = time1.split(sep=':')[0]
    day_hour2 = time2.split(sep=':')[0]
    return day_hour1==day_hour2
    
def hour_in_keys(time1,set_of_times):
    a = False,0
    for time2 in set_of_times:
        if same_hour(time1,time2):
            a= True,time2
            break
    return a

metadata_file = 'csv_metadata_14403.csv'
update_dataset = False

path_laardbeek = "/Users/grisezlrm/OneDrive - TNO/Laure Grisez - I2/Laure's copy of data/Laardbeek/"
path_schiedam = "/Users/grisezlrm/OneDrive - TNO/Laure Grisez - I2/Laure's copy of data/Schiedam/"
path_vechtstromen = "/Users/grisezlrm/OneDrive - TNO/Laure Grisez - I2/Laure's copy of data/Vechtstromen/"
path_vlaardingen = "/Users/grisezlrm/OneDrive - TNO/Laure Grisez - I2/Laure's copy of data/Vlaardingen/"
path_zuiderzeelaand = "/Users/grisezlrm/OneDrive - TNO/Laure Grisez - I2/Laure's copy of data/Zuiderzeeland/"
path_zwolle = "/Users/grisezlrm/OneDrive - TNO/Laure Grisez - I2/Laure's copy of data/Zwolle/"

dataset_path_dicts = {'Laardbeek':path_laardbeek,"Schiedam":path_schiedam,"Vechtstromen":path_vechtstromen,"Vlaardingen":path_vlaardingen,"Zuiderzeeland":path_zuiderzeelaand,"Zwolle":path_zwolle}
datasets_loaded = {'Laardbeek':"data/Laardbeek.json","Schiedam":"data\Schiedam.json","Vlaardingen":"data\Vlaardingen.json","Zuiderzeeland":"data\Zuiderzeeland.json"}


def read_dataset(Dataset,loaded_dataset=None,save_as_json=True,saved_datasets=datasets_loaded):
    if Dataset != loaded_dataset:
        update_dataset = True

    if update_dataset and (Dataset in saved_datasets.keys()): # only loads the dataset if its a new one (loading is a time comsumming process => make json maybe for the future)
        with open(datasets_loaded[Dataset], 'r') as f:
            dict_sensors_datas = json.load(f)
        loaded_dataset = Dataset
    elif update_dataset:
        path_dataset = dataset_path_dicts[Dataset]
        donnees = pd.read_csv(os.path.join(path_dataset,metadata_file),sep=";")
        dict_sensors_datas = {}
        sensors = donnees["#object_id"]
        files_names = os.listdir(path_dataset)
        for ind_sensor in range(len(sensors)) : 
            sensor = str(sensors[ind_sensor])
            for file_name in files_names:
                if sensor in file_name:
                    dir_sensor_data = os.path.join(path_dataset,file_name)
                    sensor_data = pd.read_csv(dir_sensor_data,sep=";",usecols=range(10))
                    dict_sensors_datas[sensor] = extract_sensor_data(sensor_data)
                    timestamp_info = dict_sensors_datas[sensor]["timestamp"]
                    dict_sensors_datas[sensor]["logging_start_date_time"]=timestamp_info[1]
                    dict_sensors_datas[sensor]["logging_end_date_time"]=timestamp_info[len(timestamp_info)-1]
                    dict_sensors_datas[sensor]["location_rd_x"]=float(donnees["location_rd_x"][ind_sensor])
                    dict_sensors_datas[sensor]["location_rd_y"]=float(donnees["location_rd_y"][ind_sensor])
                    # dict_sensors_datas[sensor]= {"logging_end_date_time":donnees["logging_end_date_time"][ind_sensor],"logging_start_date_time":donnees["logging_start_date_time"][ind_sensor],"location_rd_x":donnees["location_rd_x"][ind_sensor],"location_rd_y":donnees["location_rd_y"][ind_sensor]}
        if save_as_json:
            with open(os.path.join('data',Dataset+".json"), 'w') as f:
                json.dump(dict_sensors_datas, f)
        loaded_dataset = Dataset
    return dict_sensors_datas,loaded_dataset



def sensors_value_at_timestamp(time,dict_sensors):
    dict_values_at_time = {}
    for key in dict_sensors.keys():
        value_per_hour_sensor = dict_sensors[key]["data_per_hour"]
        exist,value_time = hour_in_keys(time,value_per_hour_sensor.keys()) # Finds other measures made within the same hour 
        if exist :
            dict_values_at_time[key] = value_per_hour_sensor[value_time]
            dict_values_at_time[key]['x']= dict_sensors[key]['location_rd_x']
            dict_values_at_time[key]['y']= dict_sensors[key]['location_rd_y']
    return dict_values_at_time        