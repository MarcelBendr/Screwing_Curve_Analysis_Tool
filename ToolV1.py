# -*- coding: utf-8 -*-

## Eingabebereich für den Nutzer:
#  Dateipfad zum Datensatz: (Achtung: Es müssen zwei Dateien abgelegt werden,
#  eine namens Drehmoment_df.pickle und eine namens Drehwinkel_df.pickle)
filepath = 'C:/Users/Anwender/Desktop/AURSAD/Finaler Code Masterarbeit/'
#  Soll eine Klassifikation oder Anomalieerkennung vorgenommen werden?
select_classification = False
#  Liegen Zeitreihendaten oder bereits extrahierte Features vor?
is_time_series_data = False


import pickle
import joblib  #ursprünglich: version 1.1.0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from scipy.signal import find_peaks
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import warnings

print(joblib.__version__)

with open(filepath+'Datensaetze/windowed_df2.pickle', 'rb') as input_file:
  windowed_df = pickle.load(input_file)
tqwdws = windowed_df.iloc[:, 0:1230]
label = windowed_df['label']
y_df = pd.DataFrame(label)
with open(filepath+'Datensaetze/extracted_features_filtered_torque_and_angle_intervall_left.pickle', 'rb') as input_file:
  features_filtered2_Intervall_left = pickle.load(input_file)
with open(filepath+'Datensaetze/extracted_features_filtered_torque_and_angle_intervall_right.pickle', 'rb') as input_file:
  features_filtered2_Intervall_right = pickle.load(input_file)
with open(filepath+'Datensaetze/y.pickle', 'rb') as input_file:
  label_tsfresh_interval = pickle.load(input_file)

model1_loaded = joblib.load(filepath+'Modelle/RF_Intervall_P1.joblib')
model2_loaded = joblib.load(filepath+'Modelle/RF_Intervall_P2.joblib')

def interp1d(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)

def leftover_data(dataframe, label):
    X_train_array, X_test_array, y_train_array, y_test_array = train_test_split(dataframe, label, test_size=0.2, shuffle=True, random_state=42)
    X_train_df = pd.DataFrame(X_train_array)
    y_train_df = pd.DataFrame(y_train_array)
    
    X_Train_1, X_Test_1, y_Train_1, y_Test_1 = train_test_split(X_train_df, y_train_df, test_size=0.2, shuffle=True, random_state=42)
    X_Train_1 = pd.DataFrame(X_Train_1)
    X_Test_1 = pd.DataFrame(X_Test_1)
    y_Train_1 = pd.DataFrame(y_Train_1)
    y_Test_1 = pd.DataFrame(y_Test_1)
    print(X_Train_1.shape, X_Test_1.shape, y_Train_1.shape, y_Test_1.shape)
    return X_Train_1, X_Test_1, y_Train_1, y_Test_1

def get_interval_split(X_dataframe, y_label):
    X_train_array, X_test_array, y_train_array, y_test_array = train_test_split(X_dataframe, y_label, test_size=0.2, shuffle=True, random_state=42)
    X_train_df = pd.DataFrame(X_train_array)
    X_test_df = pd.DataFrame(X_test_array)
    y_train_df = pd.DataFrame(y_train_array)
    y_test_df = pd.DataFrame(y_test_array)
    X_train_df_2, X_test_df_2, y_train_df_2, y_test_df_2 = leftover_data(X_dataframe, y_label)
    return X_train_df, X_test_df, y_train_df, y_test_df, X_train_df_2, X_test_df_2, y_train_df_2, y_test_df_2

def confusion_matrix_and_report(y_ground_truth, y_predicted):
  multiclass = confusion_matrix(y_ground_truth, y_predicted)
  fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                  colorbar=True,
                                  show_absolute=False,
                                  show_normed=True)
  plt.show()
  print(classification_report(y_ground_truth, y_predicted, digits=3))
  
def to_tsfresh_df(X_df, y_df):
  complete_torqueand_angle = []
  complete_time = []
  samples = list(range(0, X_df.shape[0]))
  sample_nr = []
  sample_list = []
  lbl_nr = []

  Messdatenlänge = X_df.iloc[0].shape[0]
  Zeitschritt = 0.01

  for i in range(X_df.shape[0]):
      
      complete_torqueand_angle.extend(X_df.iloc[i].values)
      
      timestmp = [i * Zeitschritt for i in list(range(0,Messdatenlänge))]
      complete_time.extend(timestmp)
      current_sample = samples[i]
      sample_lst = np.full(shape=X_df.iloc[i].shape[0], fill_value=1, dtype=int)
      sample_lst = sample_lst*current_sample
      sample_list.extend(sample_lst)
      sample_nr.append(current_sample)
      lbl_nr.append(y_df.iloc[i].values)
      
  complete_torqueand_angle_df = pd.DataFrame(complete_torqueand_angle)
  complete_time_df = pd.DataFrame(complete_time)
  sample_list_df = pd.DataFrame(sample_list)

  if complete_torqueand_angle_df.isnull().values.any():
    complete_torqueand_angle_df = complete_torqueand_angle_df.fillna(0)

  complete_X_df = pd.concat([sample_list_df, complete_time_df, complete_torqueand_angle_df], axis=1, ignore_index=True)
  complete_X_df.columns = ["sample_nr", "timestamp", "torque"]

  return complete_X_df
  
def extract_features_tsfresh(complete_df_for_tsfresh, y_df, X_df):
    extracted_features = extract_features(complete_df_for_tsfresh, column_id="sample_nr", column_sort="timestamp")
    extracted_features_2 = extracted_features
    impute(extracted_features_2)
    names = list(range(0, X_df.shape[0]))
    names_df = pd.DataFrame(names)
    len(names)
    new_extracted_features_2 = extracted_features_2.transpose()
    new_extracted_features_2.columns=names
    extracted_features_2 = new_extracted_features_2.transpose()
    extracted_features_2
    new_label_df = y_df.transpose()
    new_label_df.columns=names
    label_df = new_label_df.transpose()
    label_df.insert(1,'ids',names_df)
    label_df.columns = ['label', 'ids']
    y = pd.Series(label_df['label'], index=label_df.ids)
    impute(extracted_features_2)
    features_filtered_2 = select_features(extracted_features_2, y)
    return features_filtered_2

def get_suggested_labels(predictions_partial_model_1, predictions_partial_model_2):
  suggested_label = []
  for i in range(predictions_partial_model_1[0].shape[0]):
    prob1 = predictions_partial_model_1[0][i]
    prob2 = predictions_partial_model_2[0][i]

    proba_averaged = (prob1*7+prob2)/8
    suggested_avg_label = list(proba_averaged).index(max(proba_averaged))

    suggested_label.append(suggested_avg_label)
  pred_labels_combined = pd.DataFrame(suggested_label)
  return pred_labels_combined

class PartialModel1:
  def __init__(self, model1):
    self.model1 = model1

  def predict_partial_model(self, data1):
    proba1 = self.model1.predict_proba(data1)
    pred1 = self.model1.predict(data1)
    return proba1, pred1

class PartialModel2:
  def __init__(self, model2):
    self.model2 = model2

  def predict_partial_model(self, data2):
    proba2 = self.model2.predict_proba(data2)
    pred2 = self.model2.predict(data2)
    return proba2, pred2

X_train_df_left, X_test_df_left, y_train_df_left, y_test_df_left, X_train_df_2_left, X_test_df_2_left, y_train_df_2_left, y_test_df_2_left = get_interval_split(features_filtered2_Intervall_left, label_tsfresh_interval)
X_train_df_right, X_test_df_right, y_train_df_right, y_test_df_right, X_train_df_2_right, X_test_df_2_right, y_train_df_2_right, y_test_df_2_right = get_interval_split(features_filtered2_Intervall_right, label_tsfresh_interval)

if select_classification == True and is_time_series_data == False:
    Part_model1_loaded = PartialModel1(model1_loaded)
    Part_model2_loaded = PartialModel2(model2_loaded)
    predictions_model1_loaded = Part_model1_loaded.predict_partial_model(X_test_df_left.fillna(0))
    predictions_model2_loaded = Part_model2_loaded.predict_partial_model(X_test_df_right.fillna(0))
    print(classification_report(y_test_df_left, get_suggested_labels(predictions_model1_loaded, predictions_model2_loaded), digits=4))
    confusion_matrix_and_report(y_test_df_left, get_suggested_labels(predictions_model1_loaded, predictions_model2_loaded))
elif select_classification == True and is_time_series_data == True:
    with open(filepath+'Datensaetze/Drehmoment_df.pickle', 'rb') as input_file:
      Drehmoment_df = pickle.load(input_file)
    with open(filepath+'Datensaetze/Drehwinkel_df.pickle', 'rb') as input_file:
      Drehwinkel_df = pickle.load(input_file)
    with open(filepath+'Datensaetze/Label_df.pickle', 'rb') as input_file:
      Label_df = pickle.load(input_file)
    if Drehmoment_df.shape[1] != 1230:
        Arrays = []
        max_window = 1230 #window_length[window_length.idxmax()]
        for index, rows in Drehmoment_df.iterrows():
            if rows.shape[0] <= max_window:
                Verlaengertes_Array = interp1d(np.asarray(rows.values), new_len=max_window)
                Arrays.append(Verlaengertes_Array)
        Drehmoment_df = pd.DataFrame(Arrays)
        for index, rows in Drehwinkel_df.iterrows():
            if rows.shape[0] <= max_window:
                Verlaengertes_Array = interp1d(np.asarray(rows.values), new_len=max_window)
                Arrays.append(Verlaengertes_Array)
        Drehwinkel_df = pd.DataFrame(Arrays)
    complete_df_for_tsfresh = to_tsfresh_df(pd.concat([pd.DataFrame(normalize(Drehmoment_df.fillna(0), axis=1, norm='max')), pd.DataFrame(normalize(Drehwinkel_df.fillna(0), axis=1, norm='max'))], axis=0, ignore_index=True), Label_df)
    Drehmomente_Drehwinkel_extracted_features = extract_features_tsfresh(complete_df_for_tsfresh, Label_df, Drehmoment_df)
    
    warnings.warn("Klassifikator muss noch trainiert werden!")


## Für die Anomalieerkennung

def binarize_autoencoder(labels):
  binary_list = []
  for index, rows in pd.DataFrame(labels).iterrows():
    # Normale Verläufe haben eine 0, ansonsten eine 1:
    if rows.values == 0:
      binary_list.append(1)
    else:
      binary_list.append(0)
  return pd.DataFrame(binary_list)

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
    # see: https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

def divide_intervals_and_return_indexes(dataframe=tqwdws):
  Intervalls_left = []
  Intervalls_right = []
  Index_storage = []
  for index, rows in dataframe.iterrows():
    data = rows.values
    #threshold = 20 #Bei einer Störfrequenz von ca. 5 Hz sind alle 20 Datenpunkte eine Amplitude zu erwarte
    #sgn.find_peaks(data, height=None, threshold=threshold, prominence=1000)
    peaks, _ = find_peaks(data, height=0, threshold=None, prominence=0.05)
    
    idx_max = np.argmax(data[peaks])
    Maximalindex = peaks[idx_max]
    Maximalwert = data[Maximalindex]
    values_left = data[0:Maximalindex]      #Falls Peak mit enthalten: values_left = data[0:Maximalindex+1]
    values_right = data[Maximalindex::]     #Falls Peak nicht enthalten: values_right = data[Maximalindex+1::]

    Intervalls_left.append(values_left)
    Intervalls_right.append(values_right)
    Index_storage.append(Maximalindex)

  Interval_left = pd.DataFrame(Intervalls_left)
  Interval_right = pd.DataFrame(Intervalls_right)
  Index_storage = pd.DataFrame(Index_storage)
  return Interval_left, Interval_right, Index_storage

with open(filepath+'Datensaetze/Drehwinkel.pickle', 'rb') as input_file:
  Drehwinkel_df = pickle.load(input_file)

def retrieve_splitting_through_indexes (dataframe=Drehwinkel_df, index_array=[]):
  i = 1
  Intervalls_left = []
  Intervalls_right = []
  for i in range(dataframe.shape[0]):
    data = dataframe.iloc[i, :].values
    idx_split = index_array[0][i]
    values_left = data[0:idx_split]
    values_right = data[idx_split::]
    Intervalls_left.append(values_left)
    Intervalls_right.append(values_right)

    i=i+1
  Interval_left = pd.DataFrame(Intervalls_left)
  Interval_right = pd.DataFrame(Intervalls_right)
  return Interval_left, Interval_right

label_binary_df = binarize_autoencoder(y_df)

def get_normal_anomalous_data(X_df, binary_label_df):
  raw_data = pd.concat([X_df, label_binary_df], axis=1, ignore_index=True)

  raw_data = raw_data.values
  # The last element contains the labels
  labels = raw_data[:, -1]

  # The other data points are the electrocadriogram data
  data = raw_data[:, 0:-1]

  train_data, test_data, train_labels, test_labels = train_test_split(
      data, labels, test_size=0.2, random_state=42
  )

  min_val = tf.reduce_min(train_data)
  max_val = tf.reduce_max(train_data)

  train_data = (train_data - min_val) / (max_val - min_val)
  test_data = (test_data - min_val) / (max_val - min_val)

  train_data = tf.cast(train_data, tf.float32)
  test_data = tf.cast(test_data, tf.float32)

  train_labels = train_labels.astype(bool)
  test_labels = test_labels.astype(bool)

  normal_train_data = train_data[train_labels]
  normal_test_data = test_data[test_labels]

  anomalous_train_data = train_data[~train_labels]
  anomalous_test_data = test_data[~test_labels]
  return train_data, test_data, train_labels, test_labels, normal_train_data, normal_test_data, anomalous_train_data, anomalous_test_data

def get_autoencoder_anomaly_prediction(ae_model, train_data, test_data, normal_train_data, normal_test_data, anomalous_train_data, anomalous_test_data):
    encoded_data = ae_model.encoder(normal_test_data).numpy()
    decoded_data = ae_model.decoder(encoded_data).numpy()
    encoded_data = ae_model.encoder(anomalous_test_data).numpy()
    decoded_data = ae_model.decoder(encoded_data).numpy()
    reconstructions = ae_model.predict(normal_train_data)
    train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
    threshold = np.mean(train_loss) + np.std(train_loss)
    print("Threshold: ", threshold)
    reconstructions = ae_model.predict(anomalous_test_data)
    test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
    preds = predict(ae_model, test_data, threshold)
    #rint_stats(preds_right, test_labels_right)
    return preds

def get_suggested_labels_anomaly_detection(predictions_partial_model_1, predictions_partial_model_2):
  suggested_label = []
  for i in range(predictions_partial_model_1[0].shape[0]):
    pred1 = predictions_partial_model_1.iloc[i].values
    pred2 = predictions_partial_model_2.iloc[i].values
    if pred1==pred2==0:
      suggested_label.append(0)
    elif pred1==pred2==1:
      suggested_label.append(1)
    elif pred1==0 or pred2==0:
      suggested_label.append(0)
  pred_labels_combined = pd.DataFrame(suggested_label)
  return pred_labels_combined

def switch_labels(labl_df):
    if type(labl_df)==np.ndarray:
        labl_df = pd.DataFrame(labl_df)
        labl_df = labl_df.replace([0, 1], [1, 0])
        return labl_df
    else:
        labl_df = labl_df.replace([0, 1], [1, 0])
        return labl_df
    #for i in range(labl_df.shape[0]):
    #    if labl_df.iloc[i].values==0:
            


if select_classification == False:
    autoencoder_left = tf.keras.models.load_model(filepath+'Modelle/autoencoder_left_AURSAD/autoencoder_left_AURSAD', custom_objects={'f1_score':f1_score})
    autoencoder_right = tf.keras.models.load_model(filepath+'Modelle/autoencoder_right_AURSAD/autoencoder_right_AURSAD', custom_objects={'f1_score':f1_score})
    
    
    torque_left, torque_right, Index_split = divide_intervals_and_return_indexes(tqwdws.fillna(0))
    torque_left=torque_left.fillna(0)
    torque_right=torque_right.fillna(0)
    
    angle_left, angle_right = retrieve_splitting_through_indexes(Drehwinkel_df.fillna(0), Index_split)
    angle_left=angle_left.fillna(0)
    angle_right=angle_right.fillna(0)
    X_combined_left = pd.concat([pd.DataFrame(normalize(torque_left, axis=1, norm='max')), pd.DataFrame(normalize(angle_left, axis=1, norm='max'))], axis=1, ignore_index=True).fillna(0)
    X_combined_right = pd.concat([pd.DataFrame(normalize(torque_right, axis=1, norm='max')), pd.DataFrame(normalize(angle_right, axis=1, norm='max'))], axis=1, ignore_index=True).fillna(0)

    train_data_left, test_data_left, train_labels_left, test_labels_left, normal_train_data_left, normal_test_data_left, anomalous_train_data_left, anomalous_test_data_left = get_normal_anomalous_data(X_combined_left, label_binary_df)
    train_data_right, test_data_right, train_labels_right, test_labels_right, normal_train_data_right, normal_test_data_right, anomalous_train_data_right, anomalous_test_data_right = get_normal_anomalous_data(X_combined_right, label_binary_df)
    
    preds_left = get_autoencoder_anomaly_prediction(autoencoder_left, train_data_left, test_data_left, normal_train_data_left, normal_test_data_left, anomalous_train_data_left, anomalous_test_data_left)
    preds_left_int = preds_left.numpy()
    preds_left_int = preds_left_int.astype(int)
    preds_right = get_autoencoder_anomaly_prediction(autoencoder_right, train_data_right, test_data_right, normal_train_data_right, normal_test_data_right, anomalous_train_data_right, anomalous_test_data_right)
    preds_right_int = preds_right.numpy()
    preds_right_int = preds_right_int.astype(int)
    #confusion_matrix_and_report(test_labels_right, preds_right)
    pred_labels_cmbnd = get_suggested_labels_anomaly_detection(pd.DataFrame(preds_left_int), pd.DataFrame(preds_right_int))

    #print(confusion_matrix_and_report(switch_labels(test_labels_right), switch_labels(pred_labels_cmbnd)))
    print(confusion_matrix_and_report(test_labels_right, pred_labels_cmbnd))