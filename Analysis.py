import numpy as np
import pandas as pd
from PIL import Image
import os
#import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn import metrics
import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


data_set=pd.read_csv("Dataset\\Test.csv")
train_data = []
train_labels = []
train_data_flattened = []
test_data = []
test_labels=[]
test_labels.append(data_set["ClassId"])
test_labels = test_labels[0]
test_data_flattened = []
classes = 43

#Retrieving the images and their labels 
for i in range(classes):
    path = "Dataset\\Train\\" + str(i)
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            train_data.append(image)
            train_labels.append(i)
        except:
            print("Error loading image")
#Converting lists into numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)

for i in train_data:
    train_data_flattened.append(i.flatten())


path = "Dataset\\Test"
images = os.listdir(path)
for i in images[:-1]:
        try:
            image = Image.open(path + '\\'+ i)
            image = image.resize((30,30))
            image = np.array(image)
            test_data.append(image)
        except:
            print("Error loading image")
#Converting lists into numpy arrays
test_data = np.array(test_data)
test_labels = np.array(test_labels)


for i in test_data:
    test_data_flattened.append(i.flatten())


'''----------------------------------------------------------------------------
                K Nearest Neighbors
----------------------------------------------------------------------------'''
classifier = KNeighborsClassifier(n_neighbors=198)
classifier.fit(train_data_flattened, train_labels)

predicted_values = classifier.predict(test_data_flattened)

#report = classification_report(test_labels,predicted_values)
#print("The report of KNN",report)
#confusion1 = confusion_matrix(test_labels,predicted_values)
#s1=sns.heatmap(confusion1,annot=True,cmap="nipy_spectral_r")
#print(s1.set_title("CONFUSION MATRIX LR"))

acc = metrics.accuracy_score(test_labels, predicted_values)
print("The accuracy with KNN: ", round(acc*100,2), "%")
#pickle.dump(classifier, open('classifierKNN', 'wb'))


'''----------------------------------------------------------------------------
                Multinomial Logistic Regression
----------------------------------------------------------------------------'''
classifier = LogisticRegression()
classifier.fit(train_data_flattened, train_labels)

predicted_values = classifier.predict(test_data_flattened)

acc = metrics.accuracy_score(test_labels, predicted_values)
print("The accuracy with LR: ", round(acc*100,2), "%")
#pickle.dump(classifier, open('classifierLR', 'wb'))


'''----------------------------------------------------------------------------
                Random Forests
----------------------------------------------------------------------------'''
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(train_data_flattened, train_labels)

predicted_values = classifier.predict(test_data_flattened)

acc = metrics.accuracy_score(test_labels, predicted_values)
print("The accuracy with RF: ", round(acc*100,2), "%")
#pickle.dump(classifier, open('classifierRF', 'wb'))


'''----------------------------------------------------------------------------
                Convolutional Neural Networks
----------------------------------------------------------------------------'''
#Converting the labels into one hot encoding
train_labels = to_categorical(train_labels, 43)
test_labels2 = to_categorical(test_labels, 43)

#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=train_data.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
history = model.fit(train_data, train_labels, batch_size=32, epochs=epochs, validation_data=(test_data, test_labels2))
    
#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

'''
#testing accuracy on test dataset
pred = model.predict(test_data)
predicted_values = np.argmax(pred,axis=1)
#Accuracy with the test data
acc = metrics.accuracy_score(test_labels, predicted_values)

print("The accuracy with CNN: ", round(acc*100,2), "%")
'''
    

'''----------------------------------------------------------------------------
                Save the best model
----------------------------------------------------------------------------'''
#model.save("cnn_model")
#pickle.dump(classifier, open('classifier', 'wb'))
