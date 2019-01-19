import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Activation, Dropout
from dataloader import load_data, prepare_timeseries, convert_to_binary
from sklearn import metrics
import tensorflow as tf

def best_f1_score(y_true, y_pred, thresholds):
  max_res = 0
  max_thr = 0
  for threshold in thresholds:
    preds = (y_pred > threshold).astype('int')
    f1_score = metrics.f1_score(y_true, preds, average='macro') 
    if f1_score > max_res:
      max_res = f1_score
      max_thr = threshold
  return max_res, max_thr

def main():
  print('-- Loading data --')
  data = load_data()
  col_means = np.nanmean(data, axis=0)
  inds = np.where(np.isnan(data))
  #data[inds] = np.take(col_means, inds[1])
  data[inds] = 0
  X, y = prepare_timeseries(convert_to_binary(data), 30)


  train_cutoff = 64
  X_train = X[:train_cutoff]
  y_train = y[:train_cutoff]
  X_test = X[train_cutoff:]
  y_test = y[train_cutoff:]

  hidden_neurons = 50
  hidden_neurons_inner = 50
  out_neurons = 30
  dropout = 0.2
  model = Sequential()
  model.add(LSTM(output_dim=hidden_neurons_inner,
                 input_dim=X.shape[2],
                 init='uniform',
                 return_sequences=True,
                 consume_less='mem'))
  model.add(Activation('relu'))
  model.add(LSTM(output_dim=hidden_neurons_inner,
                 input_dim=hidden_neurons,
                 return_sequences=False,
                 consume_less='mem'))
  model.add(Activation('relu'))
  model.add(Dense(output_dim=hidden_neurons_inner,
                  input_dim=hidden_neurons_inner))
  model.add(Activation('relu'))
  model.add(Dense(output_dim=y.shape[-1]))
  model.add(Activation('sigmoid'))
  model.compile(loss="binary_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])
  

  
  print('-- Training --')
  epochs = 0
  batch_size = 32
  history = model.fit(X_train,
                      y_train,
                      verbose=1,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_split=0.1,
                      shuffle=False)


  print('-- Predicting --')
  y_pred = model.predict(X_test, batch_size=batch_size)
  tpr, fpr, thresholds = metrics.roc_curve(y_test[-1], y_pred[-1])
  print('ROC AUC:', metrics.roc_auc_score(y_test[-1], y_pred[-1]))
  print('Best F1 Score: {} at {} threshold'.format(*best_f1_score(y_test[-1], y_pred[-1], thresholds)))
  plt.plot(tpr, fpr)
  plt.show()

  print('-- Sanity Check --')
  print('Last date in training:', X_train[-1, -1, :3])
  # Only the last item of the test set is used for testing. The rest in
  # not used in training because it overlaps with test.
  print('First date in test:', X_test[-1, 0, :3])

if __name__ == '__main__':
  main()
