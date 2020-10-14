import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
drive.mount('/content/drive')
path="/content/drive/My Drive/dataset2.csv"
dataset= pd.read_csv(path)
X= dataset.iloc[:,0:71]
y= dataset.iloc[:,71]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
from keras import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(71, activation='relu', 
kernel_initializer='random_normal', input_dim=71))
model.add(Dense(40,activation='relu', kernel_initializer='random_normal'))
model.add(Dense(40,activation='relu', kernel_initializer='random_normal'))
model.add(Dense(40,activation='relu', kernel_initializer='random_normal'))
model.add(Dense(40,activation='relu', kernel_initializer='random_normal'))
model.add(Dense(1,activation='sigmoid',
kernel_initializer='random_normal'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
hist2=model.fit(X_train,y_train,validation_data=(X_test, y_test), batch_size=10, epochs=100)
eval_model=model.evaluate(X_train, y_train)
eval_model
print(hist2.history.keys())
y_pred=model.predict(X_test)
y_pred =(y_pred>0.5)
from sklearn.metrics import precision_score ,recall_score, f1_score
print("precision score is",precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred , average='macro'))
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)
print(hist2.history.keys())
class_names=["user","Intruder"]
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=True,
                                )
plt.show()
test_acc=model.evaluate(X_test,y_test)
train_acc=model.evaluate(X_train,y_train)
print("Test Accuracy is %f"%(test_acc[1]*100))
print("Train Accuracy is %f"%(train_acc[1]*100))
plt.plot(hist2.history['accuracy'], label='train')
plt.plot(hist2.history['val_accuracy'],label='test')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.plot(hist2.history['loss'], label='train')
plt.plot(hist2.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
model.save("model2.h5")

