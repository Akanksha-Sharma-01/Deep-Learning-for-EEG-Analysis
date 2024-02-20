import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

label = np.load('/home/arnav/Desktop/Akanksha/Dataset/EEG Dataset/Rahul/label.npy')
eeg_data = np.load('/home/arnav/Desktop/Akanksha/Dataset/EEG Dataset/Rahul/data(5-95).npy')

def preprocess(eeg_data, label):
	data = []
	labels = []
	for i in range(np.shape(eeg_data)[0]):
		for j in range(11):
			data.append(eeg_data[:,j*20:200+(1+j)*20])
			labels.append(label)
	labels = np.array(labels)
	data = np.array(data)
	return data,labels
	
model = keras.models.load_model("/home/arnav/Desktop/Akanksha/40ClassTemporalTransformer.h5")
loss = 0
acc = 0

for i,j in zip(eeg_data,label):
	data, _ = preprocess(i,j)
	pred = model.predict(data)
	pred = np.unique(np.argmax(pred, axis = 1))[0]
	loss+=abs(j-pred)
	if (j-pred)==0:
		acc+=1
print(f"Total loss: {loss/len(eeg_data)} Total accuracy: {acc/len(eeg_data)}")
