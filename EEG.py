import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

# Import training and test data
# x_train and x_test data would be of the shape: 6 Channels by 'X' # of datapoints per channel by number of training/test samples
# y_train and y_test would be of the shape: 1 field "left" or "right by number of training/test samples

# (x_train, y_train),(x_test, y_test) = *load data here*

# Normalize the training data (scale each voltage reading between 0 - 1)
# That means if we only reading voltages between 0 - 144, divide each by 144
x_train = x_train / 144
x_test = x_test / 144
print(x_train.shape)
print(x_train[0].shape)

model = Sequential

model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Last layer has softmax with 2 options: either guessing "left" or "right"
model.add(Dense(2, activation='softmax'))

opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))