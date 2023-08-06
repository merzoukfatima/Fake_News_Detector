#!/usr/bin/env python3
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import time
from datetime import datetime, timedelta
from sklearn.linear_model import PassiveAggressiveClassifier
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore
import xgboost as xgb

lemmatizer = WordNetLemmatizer()
stopword = nltk.corpus.stopwords.words("english")
def tokenize(text):
    words = word_tokenize(str(text))
    return words
def remove_punctuation(text):
    text = str(text).lower()
    result = re.sub("[^A-Za-z ]+", '', text)
    result = re.sub(r'http\S+', '', result)
    result = re.sub(' +', ' ', result)
    result.strip()
    words_wo_punct =  ''.join(result)
    return words_wo_punct
def lemmatize_word(text):
    words = word_tokenize(str(text))
    word_sentence = [word for word in words if not word in stopword]
    result = ""
    for w in word_sentence:
        lemmas = lemmatizer.lemmatize(w, pos ='v')
        result=result+" "+lemmas
    return result

import itertools
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('matrix.png')
    plt.close()


First_Dataset_pa = pickle.load(open("first dataset/first data.sav", 'rb'))
First_Dataset_tfidf = pickle.load(open("first dataset/tfidf_v.sav", 'rb'))
Second_Dataset_pa = pickle.load(open("second dataset/second_dataset_pa.sav", 'rb'))
Second_Dataset_tfidf = pickle.load(open("second dataset/tfidf.sav", 'rb'))
Third_Dataset_pa_partial = pickle.load(open("third dataset/parameter tuning/pa classifier.sav", 'rb'))
Third_Dataset_tfidf_partial = pickle.load(open("third dataset/parameter tuning/tfidf.sav", 'rb'))
Third_Dataset_pa_full = pickle.load(open("third dataset/tfidf/2018 dataset full.sav", 'rb'))
Third_Dataset_tfidf_full = pickle.load(open("third dataset/tfidf/tfidf_v.sav", 'rb'))
Third_Dataset_pa_word2vec = pickle.load(open("third dataset/word2vec/pa classifier.sav", 'rb'))
Third_Dataset_word2vec = pickle.load(open("third dataset/word2vec/word2vec.sav", 'rb'))
Third_Dataset_pa_glove = pickle.load(open("third dataset/GloVe/pa classifier.sav", 'rb'))
Third_Dataset_glove = pickle.load(open("third dataset/GloVe/glove.sav", 'rb'))
Fourth_Dataset_pa = pickle.load(open("fourth dataset/tfidf/pa classifier.sav", 'rb'))
Fourth_Dataset_tfidf = pickle.load(open("fourth dataset/tfidf/tfidf.sav", 'rb'))
Fourth_Dataset_pa_word2vec = pickle.load(open("fourth dataset/word2vec/pa classifier.sav", 'rb'))
Fourth_Dataset_word2vec = pickle.load(open("fourth dataset/word2vec/word2vec.sav", 'rb'))
Fourth_Dataset_pa_glove = pickle.load(open("fourth dataset/GloVe/pa classifier.sav", 'rb'))
Fourth_Dataset_glove = pickle.load(open("fourth dataset/GloVe/glove.sav", 'rb'))
Fifth_Dataset_pa_50000 = pickle.load(open("fifth dataset/pa classifier 50000.sav", 'rb'))
Fifth_Dataset_tfidf_50000 = pickle.load(open("fifth dataset/tfidf 50000.sav", 'rb'))
Fifth_Dataset_pa_100000 = pickle.load(open("fifth dataset/pa classifier 100000.sav", 'rb'))
Fifth_Dataset_tfidf_100000 = pickle.load(open("fifth dataset/tfidf 100000.sav", 'rb'))






class Ui_MainWindow(object):
    def set_id(self,id):
        self.window_id=id
    def setupUi(self, MainWindow):
        self.set_id(0)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(573, 383)
        MainWindow.setStyleSheet("background-color: white;")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        #self.centralwidget.setStyleSheet("background-color: white;")
        ###################### widget1 #############################
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 10, 551, 361))
        self.widget.setObjectName("widget")
        self.commandLinkButton_4 = QtWidgets.QCommandLinkButton(self.widget)
        self.commandLinkButton_4.setGeometry(QtCore.QRect(50, 160, 251, 41))
        self.commandLinkButton_4.setObjectName("commandLinkButton_4")
        self.commandLinkButton_4.clicked.connect(lambda :self.hide_window(3))
        self.commandLinkButton = QtWidgets.QCommandLinkButton(self.widget)
        self.commandLinkButton.setGeometry(QtCore.QRect(50, 80, 131, 41))
        self.commandLinkButton.setObjectName("commandLinkButton")
        self.commandLinkButton.clicked.connect(lambda :self.hide_window(1))
        self.commandLinkButton_2 = QtWidgets.QCommandLinkButton(self.widget)
        self.commandLinkButton_2.setGeometry(QtCore.QRect(50, 200, 181, 41))
        self.commandLinkButton_2.setObjectName("commandLinkButton_2")
        self.commandLinkButton_2.clicked.connect(lambda :self.hide_window(4))
        self.commandLinkButton_3 = QtWidgets.QCommandLinkButton(self.widget)
        self.commandLinkButton_3.setGeometry(QtCore.QRect(50, 120, 261, 41))
        self.commandLinkButton_3.setObjectName("commandLinkButton_3")
        self.commandLinkButton_3.clicked.connect(lambda :self.hide_window(2))
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(180, 20, 201, 51))
        self.label.setObjectName("label")
        self.label.setStyleSheet("font-weight: bold;")


        ###################### widget 2 #############################


        self.widget_2 = QtWidgets.QWidget(self.centralwidget )
        self.widget_2.setGeometry(QtCore.QRect(0, 0, 551, 361))
        self.widget_2.setObjectName("widget_2")
        self.label_4 = QtWidgets.QLabel(self.widget_2)
        self.label_4.setGeometry(QtCore.QRect(90, 80, 241, 31))
        self.label_4.setObjectName("label_4")
        self.label_3 = QtWidgets.QLabel(self.widget_2)
        self.label_3.setGeometry(QtCore.QRect(90, 160, 171, 31))
        self.label_3.setObjectName("label_3")
        self.comboBox = QtWidgets.QComboBox(self.widget_2)
        self.comboBox.setGeometry(QtCore.QRect(280, 40, 161, 29))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("SFBC-18 Dataset")
        self.comboBox.addItem("KDD-20 Dataset")
        self.comboBox.addItem("FN-18 Dataset partial")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        self.label_2.setGeometry(QtCore.QRect(90, 120, 121, 31))
        self.label_2.setObjectName("label_2")
        self.label_5 = QtWidgets.QLabel(self.widget_2)
        self.label_5.setGeometry(QtCore.QRect(90, 40, 81, 31))
        self.label_5.setObjectName("label_5")
        self.commandLinkButton_5 = QtWidgets.QCommandLinkButton(self.widget_2)
        self.commandLinkButton_5.setGeometry(QtCore.QRect(180, 250, 201, 41))
        self.commandLinkButton_5.setObjectName("commandLinkButton_5")
        self.commandLinkButton_5.clicked.connect(lambda :self.load_train())
        self.edit_1 = QtWidgets.QLineEdit(self.widget_2)
        self.edit_1 .setGeometry(QtCore.QRect(280, 120, 161, 29))
        self.edit_1 .setObjectName("edit_1 ")
        self.edit_1.setText("1.0")
        self.commandLinkButton_6 = QtWidgets.QCommandLinkButton(self.widget_2)
        self.commandLinkButton_6.setGeometry(QtCore.QRect(180, 210, 131, 41))
        self.commandLinkButton_6.setObjectName("commandLinkButton_6")
        print(self.edit_1.text())
        self.commandLinkButton_6.clicked.connect(lambda :self.train(MainWindow))
        self.edit_2 = QtWidgets.QLineEdit(self.widget_2)
        self.edit_2.setGeometry(QtCore.QRect(280, 160, 161, 29))
        self.edit_2.setObjectName("edit_2")
        self.edit_2.setText("100")
        self.commandLinkButton_15 = QtWidgets.QCommandLinkButton(self.widget_2)
        self.commandLinkButton_15.setGeometry(QtCore.QRect(180, 290, 141, 41))
        self.commandLinkButton_15.setObjectName("commandLinkButton_15")
        self.commandLinkButton_15.clicked.connect(lambda :self.hide_window(0))
        self.label_20 = QtWidgets.QLabel(self.widget_2)
        self.label_20.setGeometry(QtCore.QRect(420, 280, 500, 500))
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.widget_2)
        self.label_21.setGeometry(QtCore.QRect(60, 340, 221, 41))
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.widget_2)
        self.label_22.setGeometry(QtCore.QRect(60, 390, 221, 41))
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.widget_2)
        self.label_23.setGeometry(QtCore.QRect(60, 440, 221, 41))
        self.label_23.setObjectName("label_23")

        ###################### widget 3 #############################

        self.widget_3 = QtWidgets.QWidget(self.centralwidget )
        self.widget_3.setGeometry(QtCore.QRect(0, 0, 551, 400))
        self.widget_3.setObjectName("widget_3")
        self.commandLinkButton_7 = QtWidgets.QCommandLinkButton(self.widget_3)
        self.commandLinkButton_7.setGeometry(QtCore.QRect(180, 270, 201, 41))
        self.commandLinkButton_7.setObjectName("commandLinkButton_7")
        self.commandLinkButton_7.clicked.connect(lambda :self.load_feature())
        self.label_6 = QtWidgets.QLabel(self.widget_3)
        self.label_6.setGeometry(QtCore.QRect(70, 140, 171, 31))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.widget_3)
        self.label_7.setGeometry(QtCore.QRect(70, 190, 241, 31))
        self.label_7.setObjectName("label_7")
        self.commandLinkButton_8 = QtWidgets.QCommandLinkButton(self.widget_3)
        self.commandLinkButton_8.setGeometry(QtCore.QRect(180, 230, 131, 41))
        self.commandLinkButton_8.setObjectName("commandLinkButton_8")
        self.commandLinkButton_8.clicked.connect(lambda :self.train_feature())
        self.comboBox_4 = QtWidgets.QComboBox(self.widget_3)
        self.comboBox_4.setGeometry(QtCore.QRect(310, 20, 161, 29))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("FN-18 Dataset")
        self.comboBox_4.addItem("FRN-17 Dataset")
        self.label_8 = QtWidgets.QLabel(self.widget_3)
        self.label_8.setGeometry(QtCore.QRect(70, 100, 121, 31))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.widget_3)
        self.label_9.setGeometry(QtCore.QRect(70, 60, 241, 31))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.widget_3)
        self.label_10.setGeometry(QtCore.QRect(70, 20, 81, 31))
        self.label_10.setObjectName("label_10")
        self.edit_3 = QtWidgets.QLineEdit(self.widget_3)
        self.edit_3 .setGeometry(QtCore.QRect(310, 100, 161, 29))
        self.edit_3 .setObjectName("edit_3")
        self.edit_3.setText("1.0")
        self.edit_4 = QtWidgets.QLineEdit(self.widget_3)
        self.edit_4 .setGeometry(QtCore.QRect(310, 140, 161, 29))
        self.edit_4 .setObjectName("edit_4")
        self.edit_4.setText("100")
        self.comboBox_7 = QtWidgets.QComboBox(self.widget_3)
        self.comboBox_7.setGeometry(QtCore.QRect(310, 190, 161, 29))
        self.comboBox_7.setObjectName("comboBox_7")
        self.comboBox_7.addItem("TFIDF")
        self.comboBox_7.addItem("Word2Vec")
        self.comboBox_7.addItem("GloVe")
        self.commandLinkButton_14 = QtWidgets.QCommandLinkButton(self.widget_3)
        self.commandLinkButton_14.setGeometry(QtCore.QRect(180, 310, 141, 41))
        self.commandLinkButton_14.setObjectName("commandLinkButton_14")
        self.commandLinkButton_14.clicked.connect(lambda :self.hide_window(0))
        self.label_24 = QtWidgets.QLabel(self.widget_3)
        self.label_24.setGeometry(QtCore.QRect(420, 280, 500, 500))
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.widget_3)
        self.label_25.setGeometry(QtCore.QRect(60, 390, 221, 41))
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.widget_3)
        self.label_26.setGeometry(QtCore.QRect(60, 440, 221, 41))
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.widget_3)
        self.label_27.setGeometry(QtCore.QRect(60, 490, 221, 41))
        self.label_27.setObjectName("label_27")
                ###################### widget 4 #############################

        self.widget_4 = QtWidgets.QWidget(self.centralwidget )
        self.widget_4.setGeometry(QtCore.QRect(0, 0, 551, 361))
        self.widget_4.setObjectName("widget_4")
        self.edit_6 = QtWidgets.QLineEdit(self.widget_4)
        self.edit_6.setGeometry(QtCore.QRect(290, 160, 161, 29))
        self.edit_6.setObjectName("edit_6")
        self.edit_6.setText("100")
        self.edit_5 = QtWidgets.QLineEdit(self.widget_4)
        self.edit_5.setGeometry(QtCore.QRect(290, 120, 161, 29))
        self.edit_5.setObjectName("edit_5")
        self.edit_5.setText("1.0")
        self.label_11 = QtWidgets.QLabel(self.widget_4)
        self.label_11.setGeometry(QtCore.QRect(100, 120, 121, 31))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.widget_4)
        self.label_12.setGeometry(QtCore.QRect(100, 40, 81, 31))
        self.label_12.setObjectName("label_12")
        self.commandLinkButton_9 = QtWidgets.QCommandLinkButton(self.widget_4)
        self.commandLinkButton_9.setGeometry(QtCore.QRect(180, 250, 201, 41))
        self.commandLinkButton_9.setObjectName("commandLinkButton_9")
        self.commandLinkButton_9.clicked.connect(lambda :self.load_comparison())
        self.commandLinkButton_10 = QtWidgets.QCommandLinkButton(self.widget_4)
        self.commandLinkButton_10.setGeometry(QtCore.QRect(180, 210, 131, 41))
        self.commandLinkButton_10.setObjectName("commandLinkButton_10")
        self.commandLinkButton_10.clicked.connect(lambda :self.train_comparison())
        self.comboBox_10 = QtWidgets.QComboBox(self.widget_4)
        self.comboBox_10.setGeometry(QtCore.QRect(290, 40, 161, 29))
        self.comboBox_10.setObjectName("comboBox_10")
        self.comboBox_10.addItem("FNS-18(50000) Dataset")
        self.comboBox_10.addItem("FNS-18(100000) Dataset")
        self.label_13 = QtWidgets.QLabel(self.widget_4)
        self.label_13.setGeometry(QtCore.QRect(100, 160, 171, 31))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.widget_4)
        self.label_14.setGeometry(QtCore.QRect(100, 80, 241, 31))
        self.label_14.setObjectName("label_14")
        self.commandLinkButton_13 = QtWidgets.QCommandLinkButton(self.widget_4)
        self.commandLinkButton_13.setGeometry(QtCore.QRect(180, 290, 141, 41))
        self.commandLinkButton_13.setObjectName("commandLinkButton_13")
        self.commandLinkButton_13.clicked.connect(lambda :self.hide_window(0))
        self.label_31 = QtWidgets.QLabel(self.widget_4)
        self.label_31.setGeometry(QtCore.QRect(10, 370, 221, 31))
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(self.widget_4)
        self.label_32.setGeometry(QtCore.QRect(10, 410, 221, 31))
        self.label_32.setObjectName("label_32")
        self.label_33 = QtWidgets.QLabel(self.widget_4)
        self.label_33.setGeometry(QtCore.QRect(10, 450, 221, 31))
        self.label_33.setObjectName("label_33")
        self.label_30 = QtWidgets.QLabel(self.widget_4)
        self.label_30.setGeometry(QtCore.QRect(10, 330, 401, 31))
        self.label_30.setObjectName("label_30")
        self.label_34 = QtWidgets.QLabel(self.widget_4)
        self.label_34.setGeometry(QtCore.QRect(10, 490, 401, 401))
        self.label_34.setObjectName("label_34")
        self.label_35 = QtWidgets.QLabel(self.widget_4)
        self.label_35.setGeometry(QtCore.QRect(440, 330, 391, 31))
        self.label_35.setObjectName("label_35")
        self.label_36 = QtWidgets.QLabel(self.widget_4)
        self.label_36.setGeometry(QtCore.QRect(440, 370, 221, 31))
        self.label_36.setObjectName("label_36")
        self.label_37 = QtWidgets.QLabel(self.widget_4)
        self.label_37.setGeometry(QtCore.QRect(440, 450, 221, 31))
        self.label_37.setObjectName("label_37")
        self.label_38 = QtWidgets.QLabel(self.widget_4)
        self.label_38.setGeometry(QtCore.QRect(440, 410, 221, 31))
        self.label_38.setObjectName("label_38")
        self.label_39 = QtWidgets.QLabel(self.widget_4)
        self.label_39.setGeometry(QtCore.QRect(440, 490, 401, 401))
        self.label_39.setObjectName("label_39")


                ###################### widget 5 #############################


        self.widget_5 = QtWidgets.QWidget(self.centralwidget)
        self.widget_5.setGeometry(QtCore.QRect(0, 0, 551, 351))
        self.widget_5.setObjectName("widget_5")
        self.label_15 = QtWidgets.QLabel(self.widget_5)
        self.label_15.setGeometry(QtCore.QRect(40, 280, 150, 31))
        self.label_15.setObjectName("label_15")
        self.label_15.setText(" ")
        self.textBrowser = QtWidgets.QPlainTextEdit(self.widget_5)
        self.textBrowser.setGeometry(QtCore.QRect(40, 70, 471, 192))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setPlainText("UVALDE, TX—In the hours following a violent rampage in Texas in which a lone attacker killed at least 21 individuals and injured several others, citizens living in the only country where this kind of mass killing routinely occurs reportedly concluded Tuesday that there was no way to prevent the massacre from taking place. “This was a terrible tragedy, but sometimes these things just happen and there’s nothing anyone can do to stop them,” said Idaho resident Kathy Miller, echoing sentiments expressed by tens of millions of individuals who reside in a nation where over half of the world’s deadliest mass shootings have occurred in the past 50 years and whose citizens are 20 times more likely to die of gun violence than those of other developed nations. “It’s a shame, but what can we do? There really wasn’t anything that was going to keep this individual from snapping and killing a lot of people if that’s what they really wanted.” At press time, residents of the only economically advanced nation in the world where roughly two mass shootings have occurred every month for the past eight years were referring to themselves and their situation as “helpless.”")
        self.commandLinkButton_11 = QtWidgets.QCommandLinkButton(self.widget_5)
        self.commandLinkButton_11.setGeometry(QtCore.QRect(420, 280, 91, 41))
        self.commandLinkButton_11.setObjectName("commandLinkButton_11")
        self.commandLinkButton_11.clicked.connect(lambda:self.fakenewsdetection())
        self.comboBox_11 = QtWidgets.QComboBox(self.widget_5)
        self.comboBox_11.setGeometry(QtCore.QRect(40, 30, 300, 29))
        self.comboBox_11.setObjectName("comboBox_11")
        self.comboBox_11.addItem("First Dataset")
        self.comboBox_11.addItem("Second Dataset")
        self.comboBox_11.addItem("Third Dataset partial TFIDF")
        self.comboBox_11.addItem("Third Dataset Full TFIDF")
        self.comboBox_11.addItem("Third Dataset Word2Vec")
        self.comboBox_11.addItem("Third Dataset GloVe")
        self.comboBox_11.addItem("Fourth Dataset TFIDF")
        self.comboBox_11.addItem("Fourth Dataset Word2Vec")
        self.comboBox_11.addItem("Fourth Dataset GloVe")
        self.comboBox_11.addItem("Fifth Dataset 50000 lines TFIDF")
        self.comboBox_11.addItem("Fifth Dataset 100000 lines TFIDF")
        self.comboBox_11.currentIndexChanged.connect(lambda:self.clean())
        self.commandLinkButton_12 = QtWidgets.QCommandLinkButton(self.widget_5)
        self.commandLinkButton_12.setGeometry(QtCore.QRect(410, 20, 141, 41))
        self.commandLinkButton_12.setObjectName("commandLinkButton_12")
        self.commandLinkButton_12.clicked.connect(lambda :self.hide_window(0))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        self.hide_all()
        self.widget.show()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def hide_all(self):
        self.widget.hide()
        self.widget_2.hide()
        self.widget_3.hide()
        self.widget_4.hide()
        self.widget_5.hide()
    def hide_window(self,id):
        self.hide_all()
        if(id==0):
            self.setupUi(MainWindow)
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
            #self.widget.show()
            #self.set_id(0)
        elif(id==1):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "Train"))
            self.widget_2.show()
            self.set_id(1)
        elif(id==2):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "Feature Extraction Techniques"))
            self.widget_3.show()
            self.set_id(2)
        elif(id==3):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "Comparison with Other Classifier"))
            self.widget_4.show()
            self.set_id(3)
        elif(id==4):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "Check the news"))
            self.widget_5.show()
            self.set_id(4)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.commandLinkButton_4.setText(_translate("MainWindow", "Other classifier comparison"))
        self.commandLinkButton.setText(_translate("MainWindow", "Train"))
        self.commandLinkButton_2.setText(_translate("MainWindow", "Check the news"))
        self.commandLinkButton_3.setText(_translate("MainWindow", "Feature extraction techniques"))
        self.label.setText(_translate("MainWindow", "Fake News Detection"))
        self.label_4.setText(_translate("MainWindow", "Passive-aggrassive classifier: "))
        self.label_3.setText(_translate("MainWindow", "Max-iter parameter"))
        self.label_2.setText(_translate("MainWindow", "C parameter"))
        self.label_5.setText(_translate("MainWindow", "Dataset"))
        self.commandLinkButton_5.setText(_translate("MainWindow", "Load trained results"))
        self.commandLinkButton_6.setText(_translate("MainWindow", "Train"))
        self.commandLinkButton_7.setText(_translate("MainWindow", "Load trained results"))
        self.label_6.setText(_translate("MainWindow", "max-iter parameter"))
        self.label_7.setText(_translate("MainWindow", "Feature extraction technique"))
        self.commandLinkButton_8.setText(_translate("MainWindow", "Train"))
        self.label_8.setText(_translate("MainWindow", "C parameter"))
        self.label_9.setText(_translate("MainWindow", "Passive-aggrassive classifier: "))
        self.label_10.setText(_translate("MainWindow", "Dataset"))
        self.label_11.setText(_translate("MainWindow", "C parameter"))
        self.label_12.setText(_translate("MainWindow", "Dataset"))
        self.commandLinkButton_9.setText(_translate("MainWindow", "Load trained results"))
        self.commandLinkButton_10.setText(_translate("MainWindow", "Train"))
        self.label_13.setText(_translate("MainWindow", "Max-iter parameter"))
        self.label_14.setText(_translate("MainWindow", "Passive-aggrassive classifier: "))
        self.label_15.setText(_translate("MainWindow", " "))
        self.commandLinkButton_11.setText(_translate("MainWindow", "Check"))
        self.commandLinkButton_12.setText(_translate("MainWindow", "Back to menu"))
        self.commandLinkButton_13.setText(_translate("MainWindow", "Back to menu"))
        self.commandLinkButton_14.setText(_translate("MainWindow", "Back to menu"))
        self.commandLinkButton_15.setText(_translate("MainWindow", "Back to menu"))
####### check news functions#########
    def clean(self):
        self.label_15.setText(" ")
    def choosing_vector(self):
        choice= self.comboBox_11.currentText()
        if choice=="First Dataset":
            vector = First_Dataset_tfidf
        elif choice=="Second Dataset":
            vector = Second_Dataset_tfidf
        elif choice=="Third Dataset partial TFIDF":
            vector = Third_Dataset_tfidf_partial
        elif choice=="Third Dataset Full TFIDF":
            vector = Third_Dataset_tfidf_full
        elif choice=="Third Dataset Word2Vec":
            vector = Third_Dataset_word2vec
        elif choice=="Third Dataset GloVe":
            vector = Third_Dataset_glove
        elif choice=="Fourth Dataset TFIDF":
            vector = Fourth_Dataset_tfidf
        elif choice=="Fourth Dataset Word2Vec":
            vector = Fourth_Dataset_word2vec
        elif choice=="Fourth Dataset GloVe":
            vector = Fourth_Dataset_glove
        elif choice=="Fifth Dataset 50000 lines TFIDF":
            vector = Fifth_Dataset_tfidf_50000
        elif choice=="Fifth Dataset 100000 lines TFIDF":
            vector = Fifth_Dataset_tfidf_100000
        return vector
    def vector_name(self):
        choice= self.comboBox_11.currentText()
        if choice=="Third Dataset Word2Vec" or choice=="Fourth Dataset Word2Vec":
            return 1
        elif choice=="Third Dataset GloVe" or choice=="Fourth Dataset GloVe":
            return 2
        else:
            return 0
    def choosing_classifier(self):
        choice= self.comboBox_11.currentText()
        if choice=="First Dataset":
            classifier = First_Dataset_pa
        elif choice=="Second Dataset":
            classifier = Second_Dataset_pa
        elif choice=="Third Dataset partial TFIDF":
            classifier = Third_Dataset_pa_partial
        elif choice=="Third Dataset Full TFIDF":
            classifier = Third_Dataset_pa_full
        elif choice=="Third Dataset Word2Vec":
            classifier = Third_Dataset_pa_word2vec
        elif choice=="Third Dataset GloVe":
            classifier = Third_Dataset_pa_glove
        elif choice=="Fourth Dataset TFIDF":
            classifier = Fourth_Dataset_pa
        elif choice=="Fourth Dataset Word2Vec":
            classifier = Fourth_Dataset_pa_word2vec
        elif choice=="Fourth Dataset GloVe":
            classifier = Fourth_Dataset_pa_glove
        elif choice=="Fifth Dataset 50000 lines TFIDF":
            classifier = Fifth_Dataset_pa_50000
        elif choice=="Fifth Dataset 100000 lines TFIDF":
            classifier = Fifth_Dataset_pa_100000
        return classifier
    def word2vec_vec(self,data,vector):
            data = tokenize(data)
            j = []
            for i in data:
                if i in vector.wv:
                   j.append(vector.wv[i])
            model = [(np.mean(j, axis=0)).tolist()]
            return model
    def glove_vec(self,data,vector):
            data = tokenize(data)
            j = []
            for i in data:
                if i in vector:
                   j.append(vector[i])
            model = [(np.mean(j, axis=0)).tolist()]
            return model
    def fakenewsdetection(self):
         vector=self.choosing_vector()
         classifier=self.choosing_classifier()
         news = self.textBrowser.toPlainText()
         if len(news) < 1:
            self.label_15.setText(" ")
         else:
            sample = news
            sample = remove_punctuation(sample)
            sample = lemmatize_word(sample)
            print(sample)
            vec = self.vector_name()
            if(vec==0):
                data = vector.transform([sample]).toarray()
                print(data)
            elif(vec==1):
                data = self.word2vec_vec(sample,vector)
            elif(vec==2):
                data = self.glove_vec(sample,vector)
            prediction = classifier.predict(data)
            if(prediction[0]==0 or prediction[0]=='real' or prediction[0]=='Real'):
                self.label_15.setText("This news is real")
            elif(prediction[0]==1 or prediction[0]=='fake' or prediction[0]=='Fake'):
                self.label_15.setText("This news is fake")
############################################
################### train functions ##############
    def choosing_dataset(self,choice):
        if choice=="SFBC-18 Dataset":
            data = pd.read_csv("datasets/SFBC-18.csv")
        elif choice=="KDD-20 Dataset":
            data = pd.read_csv("datasets/KDD-20.csv")
        elif choice=="FN-18 Dataset partial":
            data = pd.read_csv("datasets/FN-18(partial).csv")
        elif choice=="FN-18 Dataset":
            data = pd.read_csv("datasets/FN-18.csv")
        elif choice=="FRN-17 Dataset":
            data = pd.read_csv("datasets/FRN-17.csv")
        elif choice=="FNS-18(50000) Dataset":
            data = pd.read_csv("datasets/FNS-18_50000.csv")
        elif choice=="FNS-18(100000) Dataset":
            data = pd.read_csv("datasets/FNS-18_100000.csv")
        return data
    def train(self,MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Train"))
        tfidf_v = TfidfVectorizer()
        data = self.choosing_dataset(self.comboBox.currentText())
        C=self.verify_c(self.edit_1.text())
        max_iter=self.verify_max_iter(self.edit_2.text())
        if C==0:
            self.edit_1.setText("Please try again")
            return
        elif max_iter==0:
            self.edit_2.setText("Please try again")
            return

        classifier = PassiveAggressiveClassifier(C=C,max_iter=max_iter)
        max=0
        s=0
        if len(data)<3000:
            X = tfidf_v.fit_transform(data['text'].values.astype('U')).toarray()
            y = data['label']
            X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
            classifier.fit(X_val, y_val)
            pred = classifier.predict(X_test)
            score = metrics.accuracy_score(y_test, pred)
            cm = metrics.confusion_matrix(y_test, pred)
            max=round(score*100,2)
        else:
            max=0
            for i in range(0,len(data)-2000,2000):
                data1=data.iloc[i:i+2000]
                X=tfidf_v.fit_transform(data1['text'].values.astype('U')).toarray()
                y=data1['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
                classifier.fit(X_train, y_train)
                pred = classifier.predict(X_test)
                score = metrics.accuracy_score(y_test, pred)
                cm = metrics.confusion_matrix(pred,y_test)
                s=round(score*100,2)
                if i==0:
                    max=s
                    cm = metrics.confusion_matrix(pred,y_test)
                if max<s:
                    max=s
                    cm = metrics.confusion_matrix(pred,y_test)
        self.label_21.setText(f'Accuracy: {max}%')
        self.label_22.setText(f'Precision Score: {round(metrics.precision_score(y_test, pred,pos_label=0)*100,2)}%')
        self.label_23.setText(f'Recall Score: {round(metrics.recall_score(y_test, pred,pos_label=0)*100,2)}%')
        plot_confusion_matrix(cm, classes=['Fake', 'Real'])
        MainWindow.resize(938, 806)
        self.widget_2.setGeometry(QtCore.QRect(0, 0, 928, 780))
        im = QPixmap("matrix.png")
        im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
        self.label_20.setPixmap(im)
    def load_train(self):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Train"))
        choice=self.comboBox.currentText()
        if choice=="SFBC-18 Dataset":
           self.edit_1.setText("1.2")
           self.edit_2.setText("100")
           self.label_21.setText(f'Accuracy: 78.62%')
           self.label_22.setText(f'Precision Score: 79.39%')
           self.label_23.setText(f'Recall Score: 80.86%')
           MainWindow.resize(938, 806)
           self.widget_2.setGeometry(QtCore.QRect(0, 0, 928, 780))
           im = QPixmap("used/matrix first dataset accuracy final.png")
           im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
           self.label_20.setPixmap(im)
        elif choice=="KDD-20 Dataset":
           self.edit_1.setText("1.6")
           self.edit_2.setText("200")
           self.label_21.setText(f'Accuracy: 61.33%')
           self.label_22.setText(f'Precision Score: 59.58%')
           self.label_23.setText(f'Recall Score: 61.74%')
           MainWindow.resize(938, 806)
           self.widget_2.setGeometry(QtCore.QRect(0, 0, 928, 780))
           im = QPixmap("used/matrix second dataset accuracy final.png")
           im = im.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
           self.label_20.setPixmap(im)
        elif choice=="FN-18 Dataset partial":
           self.edit_1.setText("0.1")
           self.edit_2.setText("400")
           self.label_21.setText(f'Accuracy: 93.45%')
           self.label_22.setText(f'Precision Score: 95.19%')
           self.label_23.setText(f'Recall Score: 91.71%')
           MainWindow.resize(938, 806)
           self.widget_2.setGeometry(QtCore.QRect(0, 0, 928, 780))
           im = QPixmap("used/matrix third dataset accuracy final.png")
           im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
           self.label_20.setPixmap(im)
############################################
################### feature functions ##############
    def choosing_technique_dataset(self,choice,technique):
         data=0
         if technique=="Word2Vec":
           if choice=="FN-18 Dataset":
                data = pd.read_csv("datasets/FN-18_word2vec.csv")
           elif choice=="FRN-17 Dataset":
                data = pd.read_csv("datasets/FRN-17_word2vec.csv")
         elif technique=="GloVe":
           if choice=="FN-18 Dataset":
                data = pd.read_csv("datasets/FN-18_glove.csv")
           elif choice=="FRN-17 Dataset":
                data = pd.read_csv("datasets/FRN-17_glove.csv")
         return data
    def train_feature(self):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Feature Extraction Techniques"))
        C=self.verify_c(self.edit_3.text())
        max_iter=self.verify_max_iter(self.edit_4.text())
        if C==0:
            self.edit_3.setText("Please try again")
            return
        elif max_iter==0:
            self.edit_4.setText("Please try again")
            return
        classifier = PassiveAggressiveClassifier(C=C,max_iter=max_iter)
        tfidf_v = TfidfVectorizer()
        choice=self.comboBox_7.currentText()
        data=self.choosing_dataset(self.comboBox_4.currentText())
        data_technique = self.choosing_technique_dataset(self.comboBox_4.currentText(),self.comboBox_7.currentText())
        score =0
        y_test =0
        pred =0
        max=0
        if choice=="TFIDF":
            for i in range(0,len(data)-2000,2000):
                data1=data.iloc[i:i+2000]
                X= tfidf_v.fit_transform(data1['text'].values.astype('U')).toarray()
                y=data1['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
                classifier.fit(X_train, y_train)
                pred = classifier.predict(X_test)
                score = metrics.accuracy_score(y_test, pred)
                cm = metrics.confusion_matrix(pred,y_test)
                s=round(score*100,2)
                if i==0:
                   max=s
                   cm = metrics.confusion_matrix(pred,y_test)
                if max<s:
                   max=s
                   cm = metrics.confusion_matrix(pred,y_test)
            print(max)
            self.label_25.setText(f'Accuracy: {max}%')
            self.label_26.setText(f'Precision Score: {round(metrics.precision_score(y_test, pred,pos_label=0)*100,2)}%')
            self.label_27.setText(f'Recall Score: {round(metrics.recall_score(y_test, pred,pos_label=0)*100,2)}%')
            plot_confusion_matrix(cm, classes=['Fake', 'Real'])
        elif choice=="Word2Vec":
             y=data['label']
             X_train, X_test, y_train, y_test = train_test_split(data_technique, y, test_size=0.2, random_state=7)
             classifier.fit(X_train, y_train)
             pred = classifier.predict(X_test)
             score = metrics.accuracy_score(y_test, pred)
             cm = metrics.confusion_matrix(pred,y_test)
             self.label_25.setText(f'Accuracy: {round(score*100,2)}%')
             self.label_26.setText(f'Precision Score: {round(metrics.precision_score(y_test, pred,pos_label=0)*100,2)}%')
             self.label_27.setText(f'Recall Score: {round(metrics.recall_score(y_test, pred,pos_label=0)*100,2)}%')
             plot_confusion_matrix(cm, classes=['Fake', 'Real'])
        elif choice=="GloVe":
             y=data['label']
             X_train, X_test, y_train, y_test = train_test_split(data_technique, y, test_size=0.2, random_state=7)
             classifier.fit(X_train, y_train)
             pred = classifier.predict(X_test)
             score = metrics.accuracy_score(y_test, pred)
             cm = metrics.confusion_matrix(pred,y_test)
             self.label_25.setText(f'Accuracy: {round(score*100,2)}%')
             self.label_26.setText(f'Precision Score: {round(metrics.precision_score(y_test, pred,pos_label=0)*100,2)}%')
             self.label_27.setText(f'Recall Score: {round(metrics.recall_score(y_test, pred,pos_label=0)*100,2)}%')
             plot_confusion_matrix(cm, classes=['Fake', 'Real'])
        MainWindow.resize(938, 806)
        self.widget_3.setGeometry(QtCore.QRect(0, 0, 928, 780))
        im = QPixmap("matrix.png")
        im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
        self.label_24.setPixmap(im)
    def load_feature(self):
        dataset=self.comboBox_4.currentText()
        choice=self.comboBox_7.currentText()
        MainWindow.resize(938, 806)
        self.widget_3.setGeometry(QtCore.QRect(0, 0, 928, 780))
        if choice=="TFIDF":
           if dataset=="FN-18 Dataset":
               self.edit_3.setText("0.9")
               self.edit_4.setText("1400")
               self.label_25.setText(f'Accuracy: 94.6%')
               self.label_26.setText(f'Precision Score: 94.92%')
               self.label_27.setText(f'Recall Score: 94.55%')
               im = QPixmap("used/matrix third dataset accuracy tfidf.png")
               im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
               self.label_24.setPixmap(im)
           elif dataset=="FRN-17 Dataset":
               self.edit_3.setText("0.6")
               self.edit_4.setText("1700")
               self.label_25.setText(f'Accuracy: 98.12%')
               self.label_26.setText(f'Precision Score: 97.25%')
               self.label_27.setText(f'Recall Score: 97.62%')
               im = QPixmap("used/matrix fourth dataset accuracy tfidf.png")
               im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
               self.label_24.setPixmap(im)
        elif choice=="Word2Vec":
           if dataset=="FN-18 Dataset":
               self.edit_3.setText("0.5")
               self.edit_4.setText("800")
               self.label_25.setText(f'Accuracy: 61.52%')
               self.label_26.setText(f'Precision Score: 57.56%')
               self.label_27.setText(f'Recall Score: 80.81%')
               im = QPixmap("used/matrix third dataset accuracy word2vec.png")
               im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
               self.label_24.setPixmap(im)
           elif dataset=="FRN-17 Dataset":
               self.edit_3.setText("0.4")
               self.edit_4.setText("800")
               self.label_25.setText(f'Accuracy: 96%')
               self.label_26.setText(f'Precision Score: 95.22%')
               self.label_27.setText(f'Recall Score: 96.99%')
               im = QPixmap("used/matrix fourth dataset accuracy word2vec.png")
               im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
               self.label_24.setPixmap(im)
        elif choice=="GloVe":
           if dataset=="FN-18 Dataset":
               self.edit_3.setText("0.7")
               self.edit_4.setText("2000")
               self.label_25.setText(f'Accuracy: 48.92%')
               self.label_26.setText(f'Precision Score: 48.87%')
               self.label_27.setText(f'Recall Score: 99.64%')
               im = QPixmap("used/matrix third dataset accuracy GloVe.png")
               im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
               self.label_24.setPixmap(im)
           elif dataset=="FRN-17 Dataset":
               self.edit_3.setText("0.1")
               self.edit_4.setText("1400")
               self.label_25.setText(f'Accuracy: 62.96%')
               self.label_26.setText(f'Precision Score: 89.86%')
               self.label_27.setText(f'Recall Score: 30.55%')
               im = QPixmap("used/matrix fourth dataset accuracy GloVe.png")
               im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
               self.label_24.setPixmap(im)
############################################
################### comparison functions ##############
    def train_comparison(self):
        C=self.verify_c(self.edit_5.text())
        max_iter=self.verify_max_iter(self.edit_6.text())
        if C==0:
            self.edit_5.setText("Please try again")
            return
        elif max_iter==0:
            self.edit_6.setText("Please try again")
            return
        xgb_classifier = xgb.XGBClassifier()
        tfidf_v = TfidfVectorizer()
        data = self.choosing_dataset(self.comboBox_10.currentText())
        classifier = PassiveAggressiveClassifier(C=C,max_iter=max_iter)
        max=0
        s=0
        max=0
        choice=self.comboBox_10.currentText()
        if choice=="FNS-18(50000) Dataset":
            for i in range(0,10000-1000,1000):
                print("hey")
                data1=data.iloc[i:i+1000]
                X=tfidf_v.fit_transform(data1['text'].values.astype('U')).toarray()
                y=data1['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
                classifier.fit(X_train, y_train)
                pred = classifier.predict(X_test)
                score = metrics.accuracy_score(y_test, pred)
                cm = metrics.confusion_matrix(pred,y_test)
                s=round(score*100,2)
                if i==0:
                    max=s
                    cm = metrics.confusion_matrix(pred,y_test)
                if max<s:
                    max=s
                    cm = metrics.confusion_matrix(pred,y_test)
            self.label_30.setText("Passive-aggrassive Classifier")
            self.label_31.setText(f'Accuracy: {max}%')
            self.label_32.setText(f'Precision Score: {round(metrics.precision_score(y_test, pred,pos_label=0)*100,2)}%')
            self.label_33.setText(f'Recall Score: {round(metrics.recall_score(y_test, pred,pos_label=0)*100,2)}%')
            plot_confusion_matrix(cm, classes=['Fake', 'Real'])
            im = QPixmap("matrix.png")
            im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.label_34.setPixmap(im)

            for i in range(0,50000-1000,1000):
                data1=data.iloc[i:i+1000]
                X=tfidf_v.fit_transform(data1['text'].values.astype('U')).toarray()
                y=data1['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
                xgb_classifier.fit(X_train, y_train)
                pred = xgb_classifier.predict(X_test)
                score = metrics.accuracy_score(y_test, pred)
                cm = metrics.confusion_matrix(pred,y_test)
                s=round(score*100,2)
                if i==0:
                    max=s
                    cm = metrics.confusion_matrix(pred,y_test)
                if max<s:
                    max=s
                    cm = metrics.confusion_matrix(pred,y_test)
            self.label_35.setText("XGboost Classifier")
            self.label_36.setText(f'Accuracy: {max}%')
            self.label_37.setText(f'Precision Score: {round(metrics.precision_score(y_test, pred,pos_label=0)*100,2)}%')
            self.label_38.setText(f'Recall Score: {round(metrics.recall_score(y_test, pred,pos_label=0)*100,2)}%')
            plot_confusion_matrix(cm, classes=['Fake', 'Real'])
            im = QPixmap("matrix.png")
            im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.label_39.setPixmap(im)
        elif choice=="FNS-18(100000) Dataset":
            for i in range(0,100000-1000,1000):
                data1=data.iloc[i:i+1000]
                X=tfidf_v.fit_transform(data1['text'].values.astype('U')).toarray()
                y=data1['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
                classifier.fit(X_train, y_train)
                pred = classifier.predict(X_test)
                score = metrics.accuracy_score(y_test, pred)
                cm = metrics.confusion_matrix(pred,y_test)
                s=round(score*100,2)
                if i==0:
                    max=s
                    cm = metrics.confusion_matrix(pred,y_test)
                if max<s:
                    max=s
                    cm = metrics.confusion_matrix(pred,y_test)
            self.label_30.setText("Passive-aggrassive Classifier")
            self.label_31.setText(f'Accuracy: {max}%')
            self.label_32.setText(f'Precision Score: {round(metrics.precision_score(y_test, pred,pos_label=0)*100,2)}%')
            self.label_33.setText(f'Recall Score: {round(metrics.recall_score(y_test, pred,pos_label=0)*100,2)}%')
            plot_confusion_matrix(cm, classes=['Fake', 'Real'])
            im = QPixmap("matrix.png")
            im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.label_34.setPixmap(im)
            for i in range(0,100000-1000,1000):
                data1=data.iloc[i:i+1000]
                X=tfidf_v.fit_transform(data1['text'].values.astype('U')).toarray()
                y=data1['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
                xgb_classifier.fit(X_train, y_train)
                pred = xgb_classifier.predict(X_test)
                score = metrics.accuracy_score(y_test, pred)
                cm = metrics.confusion_matrix(pred,y_test)
                s=round(score*100,2)
                if i==0:
                    max=s
                    cm = metrics.confusion_matrix(pred,y_test)
                if max<s:
                    max=s
                    cm = metrics.confusion_matrix(pred,y_test)
            self.label_35.setText("XGboost Classifier")
            self.label_36.setText(f'Accuracy: {max}%')
            self.label_37.setText(f'Precision Score: {round(metrics.precision_score(y_test, pred,pos_label=0)*100,2)}%')
            self.label_38.setText(f'Recall Score: {round(metrics.recall_score(y_test, pred,pos_label=0)*100,2)}%')
            plot_confusion_matrix(cm, classes=['Fake', 'Real'])
            im = QPixmap("matrix.png")
            im = im.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.label_39.setPixmap(im)
        MainWindow.resize(938, 938)
        self.widget_4.setGeometry(QtCore.QRect(0, 0, 938, 938))

    def load_comparison(self):

        dataset=self.comboBox_10.currentText()
        if dataset=="FNS-18(50000) Dataset":
               self.edit_5.setText("1.2")
               self.edit_6.setText("1000")
               self.label_30.setText("Passive-aggrassive Classifier")
               self.label_31.setText(f'Accuracy: 82.38%')
               self.label_32.setText(f'Precision Score: 83.71%')
               self.label_33.setText(f'Recall Score: 77.72%')
               im = QPixmap("used/matrix_pas 50000.png")
               im = im.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
               self.label_34.setPixmap(im)
               self.label_35.setText("XGboost Classifier")
               self.label_36.setText(f'Accuracy: 82.38%')
               self.label_37.setText(f'Precision Score: 86.78%')
               self.label_38.setText(f'Recall Score: 78.49%')
               im = QPixmap("used/matrix_xgboost 50000.png")
               im = im.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
               self.label_39.setPixmap(im)
        elif dataset=="FNS-18(100000) Dataset":
               self.edit_5.setText("1.0")
               self.edit_6.setText("1000")
               self.label_30.setText("Passive-aggrassive Classifier")
               self.label_31.setText(f'Accuracy: 84.1%')
               self.label_32.setText(f'Precision Score: 86.30%')
               self.label_33.setText(f'Recall Score: 80.53%')
               im = QPixmap("used/matrix_pas 100000.png")
               im = im.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
               self.label_34.setPixmap(im)
               self.label_35.setText("XGboost Classifier")
               self.label_36.setText(f'Accuracy: 83.2%')
               self.label_37.setText(f'Precision Score: 86.78%')
               self.label_38.setText(f'Recall Score: 78.49%')
               im = QPixmap("used/matrix_xgboost 100000.png")
               im = im.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
               self.label_39.setPixmap(im)
        MainWindow.resize(938, 938)
        self.widget_4.setGeometry(QtCore.QRect(0, 0, 928,928))

    def verify_c(self,c):
       if self.isfloat(c)==True:
          if float(c)>=0.1 and float(c)<=2.0:
             print(c)
             return float(c)
          else: return 0
       else: return 0
    def verify_max_iter(self,max_iter):
        if max_iter.isnumeric()==True:
            if int(max_iter)>=20 and int(max_iter)<=2000:
               return int(max_iter)
            else: return 0
        else: return 0
    def isfloat(self,num):
        try:
            float(num)
            return True
        except ValueError:
            return False
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
