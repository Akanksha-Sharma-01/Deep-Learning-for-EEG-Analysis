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

TrainData = eeg_data[:int(np.shape(eeg_data)[0]*0.8),:,:]
TrainLabel = label[:int(np.shape(eeg_data)[0]*0.8)]
ValData = eeg_data[int(np.shape(eeg_data)[0]*0.8):int(np.shape(eeg_data)[0]*0.9),:,:]
ValLabel = label[int(np.shape(eeg_data)[0]*0.8):int(np.shape(eeg_data)[0]*0.9)]
TestData = eeg_data[int(np.shape(eeg_data)[0]*0.9):,:,:]
TestLabel = label[int(np.shape(eeg_data)[0]*0.9):]

print(np.shape(TrainData))
print(np.shape(ValData))
print(np.shape(TestData))

TrainData, TrainLabel = preprocess(TrainData, TrainLabel)
ValData, ValLabel = preprocess(ValData, ValLabel)
TestData, TestLabel = preprocess(TestData, TestLabel)

TrainData = np.reshape(TrainData, [TrainData.shape[0], 128,220,1])
TrainLabel = np.reshape(TrainLabel, [TrainLabel.shape[0],])
ValData = np.reshape(ValData, [ValData.shape[0], 128,220,1])
ValLabel = np.reshape(ValLabel, [ValLabel.shape[0],])
TestData = np.reshape(TestData, [TestData.shape[0], 128,220,1])
TestLabel = np.reshape(TestLabel, [TestLabel.shape[0],])

print(np.shape(TrainData))
print(np.shape(ValData))
print(np.shape(TestData))

model = keras.models.load_model("/home/arnav/Desktop/Akanksha/40ClassModel1Transformer15Epoch.h5")

TrainAcc = model.evaluate(TrainData, TrainLabel)
ValAcc = model.evaluate(ValData, ValLabel)
TestAcc = model.evaluate(TestData,TestLabel)

print("Training Accuracy",TrainAcc)
print("Validation Accuracy",ValAcc)
print("Test Accuracy",TestAcc)
