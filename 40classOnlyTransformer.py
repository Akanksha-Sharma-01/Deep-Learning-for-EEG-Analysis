import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

#with tf.device('/CPU:0'):
label = np.load('/home/arnav/Desktop/Akanksha/Dataset/EEG Dataset/Rahul/label.npy')
eeg_data = np.load('/home/arnav/Desktop/Akanksha/Dataset/EEG Dataset/Rahul/data(5-95).npy')

TrainData = eeg_data[:int(np.shape(eeg_data)[0]*0.9),:,:]
TrainLabel = label[:int(np.shape(eeg_data)[0]*0.9)]
TestData = eeg_data[int(np.shape(eeg_data)[0]*0.9):,:,:]
TestLabel = label[int(np.shape(eeg_data)[0]*0.9):]

print(np.shape(TrainData))
TrainData = np.reshape(TrainData, [TrainData.shape[0], 128,440,1])
TrainLabel = np.reshape(TrainLabel, [TrainLabel.shape[0],])
TestData = np.reshape(TestData, [TestData.shape[0], 128,440,1])
TestLabel = np.reshape(TestLabel, [TestLabel.shape[0],])

maxlen = 128*440      # Only consider 3 input time points
embed_dim_1 = 128  # Features of each time point
embed_dim2 = 25
num_heads = 8   # Number of attention heads 25
ff_dim = 64     # Hidden layer size in feed forward network inside transformer

def positionalEmbedding(input,maxlen,embed_dim):
	input = layers.Reshape((maxlen,1))(input)
	emm = layers.Embedding(input_dim=maxlen+1, output_dim=embed_dim, input_length=maxlen)(input)
	emm = layers.Reshape((128,440,1))(emm)
	input = layers.Reshape((128,440,1))(input)
	output = input + emm
	return output

def transformer(input, embed_dim, rate, ff_dim):
	att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(input,input)
	att = layers.Dropout(rate)(att,training = True)
	normal1 = layers.LayerNormalization(epsilon=1e-6)(att+input)
	ff = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])(normal1)
	ff = layers.Dropout(rate)(ff, training = True)
	normal2 = layers.LayerNormalization(epsilon=1e-6)(ff+normal1)
	return normal2

def encoder(input):
	x = positionalEmbedding(input,maxlen,1)
	#x = tf.keras.layers.Reshape((128,440))(x)
	x = transformer(x, embed_dim2, 0.5, ff_dim)
	#s = np.shape(x)
	#x = tf.keras.layers.Flatten()(x)
	#x = positionalEmbedding(x,750,1)
	#x = tf.keras.layers.Reshape((30,25))(x)
	#x = transformer(x, embed_dim2, 0.5, ff_dim)
	x = tf.keras.layers.Flatten()(x)
	return x

def classifier(input):
	x = layers.Dense(100, activation="sigmoid")(input)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.5)(x)
	output = layers.Dense(40, activation="softmax")(x)
	return output

def Model():
	Input = layers.Input(shape=(128,440,1))
	Encoding = encoder(Input)
	Output = classifier(Encoding)
	Mmodel = keras.Model(inputs=Input, outputs=Output)
	return Mmodel

model = Model()
lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3,100,t_mul=2.0,m_mul=1.0,alpha=1e-5,name=None)
#my_callbacks=[keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=5 )]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),loss="SparseCategoricalCrossentropy",metrics=['accuracy'])
model.summary()
history = model.fit(TrainData, TrainLabel, batch_size=20, epochs=22, validation_split=0, validation_data=(TestData, TestLabel),verbose = 2)#, callbacks=my_callbacks)
model.save("/home/arnav/Desktop/Akanksha/40Class1Transformer.h5")

AccTr = history.history['accuracy']
LossTr = history.history['loss']
plt.subplot(2,2,1)
plt.plot(AccTr, label = 'Accuracy')
plt.plot(LossTr, label = 'Loss')
plt.legend()
plt.grid()
plt.title('Training loss and accuracy')

AccTest = history.history['val_accuracy']
LossTest = history.history['val_loss']
plt.subplot(2,2,2)
plt.plot(AccTest, label = 'Accuracy')
plt.plot(LossTest, label = 'Loss')
plt.legend()
plt.grid()
plt.title('Test loss and accuracy')

plt.subplot(2,2,3)
plt.plot(LossTr, label = 'Train')
plt.plot(LossTest, label = 'Test')
plt.legend()
plt.grid()
plt.title('Train and Test loss')

plt.subplot(2,2,4)
plt.plot(AccTr, label = 'Train')
plt.plot(AccTest, label = 'Test')
plt.legend()
plt.grid()
plt.title('Train and Test accuracy')
plt.show()

TrainAcc = model.evaluate(TrainData,TrainLabel)
TestAcc = model.evaluate(TestData,TestLabel)

print("Training Accuracy",TrainAcc)
print("Test Accuracy",TestAcc)

