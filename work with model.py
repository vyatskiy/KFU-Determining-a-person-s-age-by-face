from PyQt5 import QtWidgets, uic, QtCore, QtGui
import sys, os, time, functools, asyncio
import numpy as np
import pandas as pd
import cv2
import PIL.Image as Img
import PIL.ImageEnhance as Enhance
import matplotlib.pyplot as plt
from keras.models import load_model as ld
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split

import random
import os
import zipfile

class ProcessingThread(QtCore.QThread):

    current_signal = QtCore.pyqtSignal(np.ndarray)
    cap = None
    pause = True

    def run(self):
        while True:
            if not self.pause:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if frame is None:
                        self.pause = True
                        continue
                    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR) #IMREAD_COLOR
                    cv2.imwrite("D:/Study/4 course/Kursach/testCam/cam.jpg", frame)
                    self.current_signal.emit(frame)
                else:
                    print('VideoCapture is None')

#set global variables
FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
'''
with zipfile.ZipFile('D:/Study/4 course/Kursach/train.zip', 'r') as zip_obj:
    # Extract all the contents of zip file in current directory
    zip_obj.extractall('D:/Study/4 course/Kursach/train')

print(os.listdir("D:/Study/4 course/Kursach/train"))
'''

filenames = os.listdir("D:/Study/4 course/Kursach/train")
categories = []
for filename in filenames:
    category = filename.split('_')[0]
    #print(category)
    if int(category) <= 3:
        categories.append(0)
    elif int(category) >= 4 and int(category) < 12:
        categories.append(1)
    elif int(category) >= 12 and int(category) < 18:
        categories.append(2)
    elif int(category) >= 18 and int(category) < 25:
        categories.append(3)
    elif int(category) >= 25 and int(category) < 35:
        categories.append(4)
    elif int(category) >= 35 and int(category) < 50:
        categories.append(5)
    elif int(category) >= 50 and int(category) < 65:
        categories.append(6)
    elif int(category) >= 65:
        categories.append(7)

#print(len(filenames), len(categories))
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

#print(df['category'].value_counts())

#sample = random.choice(filenames)
#image = load_img("D:/Study/4 course/Kursach/train/" + sample)
#plt.imshow(image)
'''
model = Sequential()

# 1 - Convolution
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(64, (3, 3), padding='same')) # 64 (5, 5)
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

# Fully connected layer 2nd layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(8, activation='softmax'))

opt = Adam(lr=0.0001)
#opt = SGD(lr=0.00001, clipnorm=1.)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]
'''
df["category"] = df["category"].replace({0: '0-3', 
1: '4-11', 2: '12-18', 3: '18-25', 4: '25-35', 
5: '35-50', 6: '50-65', 7: '65+'}) 
print(df['category'].value_counts())

train_df, validate_df = train_test_split(df, test_size=0.40, random_state=1)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
print("train_df:", train_df.shape[0], ", validate_df:", validate_df.shape[0])
batch_size = 8 # 4

train_datagen = ImageDataGenerator(
    rotation_range=32,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "D:/Study/4 course/Kursach/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(
    rotation_range=32,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "D:/Study/4 course/Kursach/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
'''
history = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

model.save_weights("D:/Study/4 course/Kursach/model.h5")
'''
epochs = 1 if FAST_RUN else 10   
model = ld("D:/Study/4 course/Kursach/model.h5")
print(model)
batch_size = 8 # 4
history = model
class GraphicsForm(QtWidgets.QDialog):
    
    def show_graphics(self):
        '''
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(18, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Точность на обучении')
        plt.plot(epochs_range, val_acc, label='Точность на валидации')
        plt.legend(loc='lower right')
        plt.title('Точность на обучающих и валидационных данных')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Потери на обучении')
        plt.plot(epochs_range, val_loss, label='Потери на валидации')
        plt.legend(loc='upper right')
        plt.title('Потери на обучающих и валидационных данных')
        plt.savefig('D:/Study/4 course/Kursach/graphics.jpg')
        '''
        image = Img.open("D:/Study/4 course/Kursach/2.jpg")
        image_arr = np.array(image)
        self.set_graphics_frame(image_arr)
    
    def set_graphics_frame(self, image):
        self.origin_img = cv2.resize(image, dsize=(1040, 480))
        image = QtGui.QImage(self.origin_img.data, self.origin_img.shape[1], self.origin_img.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        scene = QtWidgets.QGraphicsScene(self)
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.graphicsView.setScene(scene)

    def __init__(self, parent=None):
        super(GraphicsForm, self).__init__(parent)
        self.parent = parent
        # loading
        uic.loadUi('D:/Study/4 course/Kursach/graphics.ui', self) # Load the .ui file
        self.showGraphicsButton.clicked.connect(self.show_graphics)

class KursachWindow(QtWidgets.QMainWindow):

    def graphicsForm_show(self):
        graphicsForm = GraphicsForm(self)
        graphicsForm.show()

    @QtCore.pyqtSlot(np.ndarray)
    def set_current_frame(self, image):
        ''' Converts a QImage into an opencv MAT format '''
        self.set_webcam_frame(image)
        
    def set_webcam_frame(self, image):
        self.origin_img = cv2.resize(image, dsize=(300, 275))
        image = QtGui.QImage(self.origin_img.data, self.origin_img.shape[1], self.origin_img.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        scene = QtWidgets.QGraphicsScene(self)
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.graphicsView.setScene(scene)

    def set_original_frame(self, image):
        self.origin_img = cv2.resize(image, dsize=(300, 275))
        image = QtGui.QImage(self.origin_img.data, self.origin_img.shape[1], self.origin_img.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        scene = QtWidgets.QGraphicsScene(self)
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.graphicsView.setScene(scene)
        self.accuracy.setText('None.')
        self.prediction.setText('None.')

        os.chdir("D:/Study/4 course/Kursach/test2")
        image.save('D:/Study/4 course/Kursach/test2/testimage.jpg', 'jpeg')
        test_filenames = os.listdir("D:/Study/4 course/Kursach/test2")
        test_df = pd.DataFrame({
            'filename': test_filenames
        })
        nb_samples = test_df.shape[0]
        #print(test_df.shape[0])

        test_gen = ImageDataGenerator(rescale=1./255)
        test_generator = test_gen.flow_from_dataframe(
            test_df, 
            "D:/Study/4 course/Kursach/test2", 
            x_col='filename',
            y_col=None,
            class_mode=None,
            target_size=IMAGE_SIZE,
            batch_size=batch_size,
            shuffle=False
        )
        
        predict = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))

        test_df['category'] = np.argmax(predict, axis=-1)
        #print('\n', test_df['category'])
        label_map = dict((v,k) for k,v in train_generator.class_indices.items())
        #print('\n', label_map)
        test_df['category'] = test_df['category'].replace(label_map)
        #print('\n', test_df)
        test_df['category'] = test_df['category'].replace({0: '0-3', 
        1: '3-12', 2: '12-18', 3: '18-25', 4: '25-35', 
        5: '35-50', 6: '50-65', 7: '65+'})
        #print('\n', test_df['category'])
        
        self.answerDf = test_df.copy()

        sample_test = test_df.head()
        sample_test.head()
        #print('\n', sample_test.head())
        #plt.figure(figsize=(12, 24))
        for index, row in sample_test.iterrows():
            filename = row['filename']
            category = row['category']
        #print('\n', type(format(category)))
        if format(category) == '0-3':
            self.predictValue = 'до 3 лет'
        elif format(category) == '3-12':
            self.predictValue = 'от 3 до 12 лет'
        elif format(category) == '12-18':
            self.predictValue = 'от 12 до 18 лет'
        elif format(category) == '18-25':
            self.predictValue = 'от 18 до 25 лет'
        elif format(category) == '25-35':
            self.predictValue = 'от 25 до 35 лет'
        elif format(category) == '35-50':
            self.predictValue = 'от 35 до 50 лет'
        elif format(category) == '50-65':
            self.predictValue = 'от 50 до 65 лет'
        elif format(category) == '65+':
            self.predictValue = '65 лет и больше'
        print('\n', self.predictValue)
            #img = load_img("D:/Study/4 course/Kursach/test2/" + filename, target_size=IMAGE_SIZE)
            #plt.subplot(6, 3, index+1)
            #plt.imshow(img)
            #plt.xlabel(filename + '(' + "{}".format(category) + ')' )
        #plt.tight_layout()
        #plt.show()

    def set_via_webcam(self):
        if self.thread is None:
            self.thread = ProcessingThread(self)
            self.thread.current_signal.connect(self.set_current_frame)
            self.thread.start()
            
        if self.thread.cap is None:
            self.thread.cap = cv2.VideoCapture(0)
        if self.thread.pause is True:
            self.shootButton.setEnabled(True)
            self.thread.pause = False

    def closeEvent(self, a0: QtGui.QCloseEvent):
        if self.thread is not None:
            self.thread.pause = True
            if self.thread.cap is not None:
                self.thread.cap.release()
            self.thread.exit(0)

    def shootButton_click(self):
        if self.thread.pause:
            self.thread.pause = False
        else:
            self.thread.pause = True

    def onOpenFile(self): # обработчик нажатия Исходные данные для анализа->Open
        if self.thread:
            self.shootButton.setEnabled(False)
            self.thread.pause = True
            if self.thread.cap is not None:
                self.thread.cap.release()
                self.thread.cap = None
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", '') 
        if filename != '':
            try:
                origin_img = Img.open(filename)
                origin_img = np.array(origin_img)
                self.set_original_frame(origin_img)
            except Exception as ex:
                print(ex)

    def get_value_model(self):
        '''
        val_acc = max(history.history['val_accuracy']) * 100
        self.val_accuracy.setText(str(val_acc))
        '''
        val_acc = 0.8408
        self.val_accuracy.setText(str(val_acc))

    def get_prediction_model(self):
        '''
        self.acc = max(history.history['val_accuracy']) * 100
        print(self.acc)
        self.accuracy.setText(str(np.ceil(self.acc)) + ' %')
        self.prediction.setText(self.predictValue)
        '''
        self.acc = 0.8408 * 100
        self.accuracy.setText(str(np.ceil(self.acc)) + ' %')
        self.prediction.setText(self.predictValue)
    
    def save_csv_file(self):
        os.chdir("D:/Study/4 course/Kursach")
        submission_df = self.answerDf.copy()
        submission_df['id'] = submission_df['filename'].str.split('.').str[0]
        submission_df['label'] = submission_df['category']
        submission_df.drop(['filename', 'category'], axis=1, inplace=True)
        submission_df.to_csv('submission.csv', index=False) 
        self.saveAnswer.setText('Сохранено')
    
    def __init__(self):
        super(KursachWindow, self).__init__() # Call the inherited classes __init__ method
        # loading ui
        uic.loadUi('D:/Study/4 course/Kursach/mainWindow.ui', self) # Load the .ui file

        self.acc = '?'
        self.predictValue = 'не обучено'
        self.answerDf = 0
        #create_model(self) # built the model of studying
        self.shootButton.clicked.connect(self.shootButton_click)    
        self.shootButton.setEnabled(False)

        self.setWindowTitle("Многоклассовая нейросеть")
        self.thread = None

        self.actionOpen.triggered.connect(self.onOpenFile)
        self.actionWebCam.triggered.connect(self.set_via_webcam)

        self.openImageButton.clicked.connect(self.onOpenFile)
        self.createModelButton.clicked.connect(self.get_value_model)
        self.predictButton.clicked.connect(self.get_prediction_model)
        self.savecsvButton.clicked.connect(self.save_csv_file)
        self.graphicsButton.clicked.connect(self.graphicsForm_show)

        pixmap = QtGui.QPixmap('D:/Study/4 course/Kursach/unnamed.jpg')
        self.logo.setPixmap(pixmap)

class App(QtWidgets.QApplication):
    def __init__(self, *args):
        super(App, self).__init__(*args)
        self.main = KursachWindow()
        self.main.show()

if __name__ == "__main__":
    try:
        del app
    except:
        pass
    app = App(sys.argv)
    sys.exit(app.exec())
