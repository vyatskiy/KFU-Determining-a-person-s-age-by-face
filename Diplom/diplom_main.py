from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5 import *
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
from tensorflow.keras.preprocessing import image

import random
import os
import signal
import zipfile
import detectAgeAndGenderCaffeNet

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
                    cv2.imwrite("testCam/cam.jpg", frame)
                    self.current_signal.emit(frame)
                else:
                    print('VideoCapture is None')

#set global variables
FAST_RUN = False
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
'''
with zipfile.ZipFile('train.zip', 'r') as zip_obj:
    # Extract all the contents of zip file in current directory
    zip_obj.extractall('train')

print(os.listdir("train"))
'''

filenames = os.listdir("train")
categories = []
for filename in filenames:
    category = filename.split('_')[0]
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

#   try to add second dataset to train

# imgNames = []
# filenamesForSecondData = os.listdir("train2")
# # print(filenamesForSecondData)
# for filename in filenamesForSecondData:
#     category = int(filename)
#     print(category)
#     if category <= 3:
#         imageNameForSecondData = os.listdir("train2/" + filename)
#         for imagename in imageNameForSecondData:
#             imgNames.append(imagename)
#             categories.append(0)
#     elif category >= 4 and category < 12:
#         imageNameForSecondData = os.listdir("train2/" + filename)
#         for imagename in imageNameForSecondData:
#             imgNames.append(imagename)
#             categories.append(1)
#     elif category >= 12 and category < 18:
#         imageNameForSecondData = os.listdir("train2/" + filename)
#         for imagename in imageNameForSecondData:
#             imgNames.append(imagename)
#             categories.append(2)
#     elif category >= 18 and category < 25:
#         imageNameForSecondData = os.listdir("train2/" + filename)
#         for imagename in imageNameForSecondData:
#             imgNames.append(imagename)
#             categories.append(3)
#     elif category >= 25 and category < 35:
#         imageNameForSecondData = os.listdir("train2/" + filename)
#         for imagename in imageNameForSecondData:
#             imgNames.append(imagename)
#             categories.append(4)
#     elif category >= 35 and category < 50:
#         imageNameForSecondData = os.listdir("train2/" + filename)
#         for imagename in imageNameForSecondData:
#             imgNames.append(imagename)
#             categories.append(5)
#     elif category >= 50 and category < 65:
#         imageNameForSecondData = os.listdir("train2/" + filename)
#         for imagename in imageNameForSecondData:
#             imgNames.append(imagename)
#             categories.append(6)
#     elif category >= 65:
#         imageNameForSecondData = os.listdir("train2/" + filename)
#         for imagename in imageNameForSecondData:
#             imgNames.append(imagename)
#             categories.append(7)   

# end trying to add second dataset to train
filenames = os.listdir("train") # + imgNames

print(len(filenames), len(categories))
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

'''
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=8, activation="softmax"))

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
'''
earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

df["category"] = df["category"].replace({0: '0-3', 
1: '4-11', 2: '12-18', 3: '18-25', 4: '25-35', 
5: '35-50', 6: '50-65', 7: '65+'}) 
print(df['category'].value_counts())

train_df, validate_df = train_test_split(df, test_size=0.10, random_state=1)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
print("train_df:", train_df.shape[0], ", validate_df:", validate_df.shape[0])
batch_size = 16 # 8

train_datagen = ImageDataGenerator( # аугментация
    rotation_range=40,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
'''
epochs = 1 if FAST_RUN else 50

history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
#     validation_steps=total_validate//batch_size,
#     steps_per_epoch=total_train//batch_size,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=callbacks
)


model.save_weights("model.h5")
'''
model = ld("model16.h5")
print(model)
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
        plt.savefig('graphics.jpg')
        '''
        image = Img.open("graphics.jpg")
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
        uic.loadUi('graphics.ui', self) # Load the .ui file
        self.showGraphicsButton.clicked.connect(self.show_graphics)
        self.setWindowTitle("Многоклассовая нейросеть - Graphics")
        QtWidgets.QDialog.setStyleSheet(self, "border-image: url(background.jpg);")

class InfoVGG16Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(InfoVGG16Form, self).__init__(parent)
        self.parent = parent
        uic.loadUi('info_about_VGG16.ui', self) # Load the .ui file

        self.model.setStyleSheet("border-image: url(vgg16.png);")
        self.architecture.setStyleSheet("border-image: url(vgg16_2.png);")
        self.table.setStyleSheet("border-image: url(vgg16_3.jpg);")
        self.setWindowTitle("Многоклассовая нейросеть - Information")
        QtWidgets.QDialog.setStyleSheet(self, "border-image: url(background.jpg);")

class VGG16ModelWindow(QtWidgets.QMainWindow):

    def graphicsForm_show(self):
        graphicsForm = GraphicsForm(self)
        graphicsForm.show()
    
    def infoForm_show(self):
        infoForm = InfoVGG16Form(self)
        infoForm.show()

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

        image.save('test2/testimage.jpg', 'jpeg')
        test_filenames = os.listdir("test2")
        test_df = pd.DataFrame({
            'filename': test_filenames
        })
        nb_samples = test_df.shape[0]

        test_gen = ImageDataGenerator(rescale=1./255)
        test_generator = test_gen.flow_from_dataframe(
            test_df, 
            "test2", 
            x_col='filename',
            y_col=None,
            class_mode=None,
            target_size=IMAGE_SIZE,
            batch_size=batch_size,
            shuffle=False
        )
        
        predict = model.predict(test_generator, steps=np.ceil(nb_samples/batch_size))

        test_df['category'] = np.argmax(predict, axis=-1)
        label_map = dict((v,k) for k,v in train_generator.class_indices.items())
        test_df['category'] = test_df['category'].replace(label_map)
        test_df['category'] = test_df['category'].replace({0: '0-3', 
        1: '3-12', 2: '12-18', 3: '18-25', 4: '25-35', 
        5: '35-50', 6: '50-65', 7: '65+'})
        
        self.answerDf = test_df.copy()
        sample_test = test_df.head()
        sample_test.head()

        for index, row in sample_test.iterrows():
            filename = row['filename']
            category = row['category']

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
        else:
            os.kill (os.getpid (), signal.SIGTERM)

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
        inputName = filename.split('/')
        print(str(inputName[-1:]))
        if filename != '':
            try:
                origin_img = Img.open(filename)
                origin_img = np.array(origin_img)
                self.set_original_frame(origin_img)
            except Exception as ex:
                print(ex)

    def get_value_model(self):
        # val_acc = max(history.history['val_accuracy']) * 100
        # self.val_accuracy.setText(str(val_acc))
 
        val_acc = 0.621
        self.val_accuracy.setText(str(val_acc))

    def get_prediction_model(self):
        # self.acc = max(history.history['val_accuracy']) * 100
        # print(self.acc)
        # self.accuracy.setText(str(np.ceil(self.acc)) + ' %')
        # self.prediction.setText(self.predictValue)
        
        self.acc = 0.621 * 100
        self.accuracy.setText(str(np.ceil(self.acc)) + ' %')
        self.prediction.setText(self.predictValue)
    
    def save_csv_file(self):
        submission_df = self.answerDf.copy()
        submission_df['id'] = submission_df['filename'].str.split('.').str[0]
        submission_df['label'] = submission_df['category']
        submission_df.drop(['filename', 'category'], axis=1, inplace=True)
        submission_df.to_csv('submission.csv', index=False) 
        self.saveAnswer.setText('Сохранено')

    def __init__(self):
        super(VGG16ModelWindow, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('vgg16.ui', self) # Load the .ui file

        self.acc = '?'
        self.predictValue = 'не обучено'
        self.answerDf = 0
        #create_model(self) # built the model of studying
        self.shootButton.clicked.connect(self.shootButton_click)    
        self.shootButton.setEnabled(False)

        self.thread = None

        self.actionOpen.triggered.connect(self.onOpenFile)
        self.actionWebCam.triggered.connect(self.set_via_webcam)

        self.openImageButton.clicked.connect(self.onOpenFile)
        self.createModelButton.clicked.connect(self.get_value_model)
        self.predictButton.clicked.connect(self.get_prediction_model)
        self.savecsvButton.clicked.connect(self.save_csv_file)
        self.graphicsButton.clicked.connect(self.graphicsForm_show)
        self.infoAboutVGG16Button.clicked.connect(self.infoForm_show)

        pixmap = QtGui.QPixmap('unnamed.jpg')
        self.logo.setPixmap(pixmap)
        
        self.setWindowTitle("Многоклассовая нейросеть - VGG_16")
        QtWidgets.QMainWindow.setStyleSheet(self, "border-image: url(back.jpg);")

class InfoCaffeNetForm(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(InfoCaffeNetForm, self).__init__(parent)
        self.parent = parent
        uic.loadUi('info_about_CaffeNet.ui', self) # Load the .ui file

        self.accuracy.setStyleSheet("border-image: url(caffeNet accuracy 2.png);")
        self.results.setStyleSheet("border-image: url(caffeNet accuracy.png);")
        self.graphics.setStyleSheet("border-image: url(graphics age.png);")
        self.setWindowTitle("Многоклассовая нейросеть - Information")
        QtWidgets.QDialog.setStyleSheet(self, "border-image: url(background.jpg);")

class CaffeModelWindow(QtWidgets.QMainWindow):

    def infoForm_show(self):
        infoForm = InfoCaffeNetForm(self)
        infoForm.show()

    def onOpenFile(self): # обработчик нажатия Исходные данные для анализа->Open
        if self.thread:
            self.shootButton.setEnabled(False)
            self.thread.pause = True
            if self.thread.cap is not None:
                self.thread.cap.release()
                self.thread.cap = None
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", '') 
        self.input.setText(filename)
        self.executeOpenCV()

    def closeEvent(self, a0: QtGui.QCloseEvent):
        if self.thread is not None:
            self.thread.pause = True
            if self.thread.cap is not None:
                self.thread.cap.release()
            self.thread.exit(0)
        else:
            os.kill (os.getpid (), signal.SIGTERM)

    def executeOpenCV(self):
        if os.path.exists(self.input.text()):
            detectAgeAndGenderCaffeNet.setImagePath(self.input.text())
            detectAgeAndGenderCaffeNet.detect()
        elif self.input.text() == "":
            detectAgeAndGenderCaffeNet.setImagePath(self.input.text())
            detectAgeAndGenderCaffeNet.detect()
        else:
            QtWidgets.QMessageBox.critical(self, "Error!", "FileNotFoundExceptoin !\n"
            "Enter a correct image's name in this directory\n"
            "Maybe, inputFile is not exists or locate to another dir")

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('caffenet.ui', self) # Load the .ui file

        self.thread = None
        self.actionOpen.triggered.connect(self.onOpenFile)
        self.launchButton.clicked.connect(self.executeOpenCV)
        self.infoCaffeNetButton.clicked.connect(self.infoForm_show)

        pixmap = QtGui.QPixmap('unnamed.jpg')
        self.logo.setPixmap(pixmap)

        self.setWindowTitle("Многоклассовая нейросеть - CaffeNet")
        QtWidgets.QMainWindow.setStyleSheet(self, "border-image: url(back.jpg);")

class StartWindow(QtWidgets.QMainWindow):
    def open_main_window(self):
        if self.VGGRadioButton.isChecked():
            self.model_1 = VGG16ModelWindow()
            self.model_1.show()
        if self.CaffeRadioButton.isChecked():
            self.model_2 = CaffeModelWindow()
            self.model_2.show()

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('start.ui', self) # Load the .ui file

        self.startMainButton.clicked.connect(self.open_main_window)

        self.btngroup1 = QtWidgets.QButtonGroup()
        self.btngroup1.addButton(self.VGGRadioButton)
        self.btngroup1.addButton(self.CaffeRadioButton)
        self.btngroup1.setExclusive(True)

        self.setWindowTitle("Многоклассовая нейросеть - Start")
        QtWidgets.QMainWindow.setStyleSheet(self, "border-image: url(back.jpg);")
            
class App(QtWidgets.QApplication):
    def __init__(self, *args):
        super(App, self).__init__(*args)
        self.start = StartWindow()
        self.start.show()

if __name__ == "__main__":
    try:
        del app
    except:
        pass
    app = App(sys.argv)
    sys.exit(app.exec())
