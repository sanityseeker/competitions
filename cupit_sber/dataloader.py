import numpy as np
import os
import sys
from collections import defaultdict


def load_initial_data():
  data_path = 'data/trade_data.csv'
  data = None
  if os.path.exists(data_path):
    data = np.loadtxt(data_path, dtype='object', delimiter=',', skiprows=1)
  else:
    print("trade_data.csv is not in data folder", file=sys.stderr)
    return data
  
  dates = list(map(lambda x: tuple(map(int, x.split('.'))), data[:, 0]))
  values = data[:, 5:6].astype('float')
  result = defaultdict(list)
  for date, value in zip(dates, values):
    result[date].append(value)
  return result, dates


def load_data():
  initial_data, dates = load_initial_data()
  indicators = list(load_indicators())
  
  if not initial_data:
    return

  result_list = []
  for date in dates:
   result_list.append(list(date) + initial_data[date])
   for indicator, max_len in indicators:
     ind_values = indicator[date]
     if len(ind_values) < max_len:
       ind_values.extend([np.nan for i in range(max_len - len(ind_values))])
     result_list[-1].extend(ind_values)
   result_list[-1] = np.array(result_list[-1])
 

  result = np.vstack(result_list)
  return result


def convert_date_from_indicator(date):
  date_list = date.split('-')[::-1]
  date_list[2] = date_list[2][2:]
  return tuple(map(int, date_list))


def load_indicators():
  for i in range(10):
    data_path = 'data/indicator{}.csv'.format(i)
    max_len = 0
    
    data = defaultdict(list)
    if os.path.exists(data_path):
      ind_data = np.loadtxt(data_path, dtype='object', delimiter=',')
      dates = map(convert_date_from_indicator, ind_data[:, 0])
      for i, date in enumerate(dates):
        data[date] = ind_data[i, 1:].astype('float')
        if len(data[date]) > max_len:
          max_len = len(data[date])
    else:
      print('No indicator{}.csv in data'.format(i))
    yield data, max_len

      


def convert_to_binary(data):
  diffs = data[1:, 3] - data[:-1, 3]
  data[0, 3] = 0
  data[1:, 3] = diffs > 0
  return data

def prepare_timeseries(data, length):
  series_list = []
  targets_list = []
  for start in range(len(data) - 2*length + 1):
    series_list.append(data[start:start+length])
    targets_list.append(data[start+length:start+2*length, 3])

  X = np.array(series_list)
  y = np.vstack(targets_list)
  return X, y

  


  
