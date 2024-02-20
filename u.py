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
			data.append(eeg_data[i,:,j*20:200+(1+j)*20])
			labels.append(label[i])
	labels = np.array(labels)
	data = np.array(data)
	return data,labels

TrainData = eeg_data[:int(np.shape(eeg_data)[0]*0.9),:,:]
TrainLabel = label[:int(np.shape(eeg_data)[0]*0.9)]
TestData = eeg_data[int(np.shape(eeg_data)[0]*0.9):,:,:]
TestLabel = label[int(np.shape(eeg_data)[0]*0.9):]

TrainData, TrainLabel = preprocess(TrainData, TrainLabel)
TestData, TestLabel = preprocess(TestData, TestLabel)

TrainData = np.reshape(TrainData, [TrainData.shape[0], 128,220,1])
TrainLabel = np.reshape(TrainLabel, [TrainLabel.shape[0],])
TestData = np.reshape(TestData, [TestData.shape[0], 128,220,1])
TestLabel = np.reshape(TestLabel, [TestLabel.shape[0],])

print(np.max(TrainData))
print(np.max(TestData))

#model = keras.models.load_model("/home/arnav/Desktop/Akanksha/40ClassModel1Bi_LSTM22.h5")

#TrainAcc = model.evaluate(TrainData, TrainLabel)
#TestAcc = model.evaluate(TestData,TestLabel)

#print("Training Accuracy",TrainAcc)
#print("Test Accuracy",TestAcc)
