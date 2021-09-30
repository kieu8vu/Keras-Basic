# Importing Keras Sequential Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import numpy

from tensorflow.keras.models import load_model
# load model weights, but do not compile

#1.Load Dữ liệu và chia Train, Val và Test
from numpy import loadtxt
from sklearn.model_selection import train_test_split
dataset = numpy.loadtxt('datasets/pima-indians-diabetes.csv', delimiter=",")

X= dataset[:,0:8]
y= dataset[:,8]
X_train_val, X_test, y_train_val, y_test =train_test_split(X,y,test_size=0.2)
X_train, X_val, y_train, y_val =train_test_split(X_train_val,y_train_val,test_size=0.2)

#viet model (tao model)
# model= Sequential()
# model.add(Dense(16, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

# #compile model
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# #train model
# model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_val , y_val))

# #luu model
# model.save("mymodel.h5")

#load model
model=load_model("mymodel.h5")

loss, acc=model.evaluate(X_test,y_test)
print("Loss=",loss)
print("Acc=",acc)

X_new= X_test[10]
y_new=y_test[10]

X_new=numpy.expand_dims(X_new,axis=0)

y_predict=model.predict(X_new)

result ="Tieu duong (1)"
if y_predict<=0.5:
	result="Khong tieu duong (0)"
print("Gia tri du doan= ",result)
print("Gia tri dung= ",y_new)
print(X_new)