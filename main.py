import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from PIL import Image
import PIL.ImageOps
import os


X,y = fetch_openml('mnist_784',version=1,return_X_y=True)
classes = ['0','1','2','3','4','5','6','7','8','9']
print(pd.Series(y).value_counts())
nClasses = len(classes)

xtrain,xtest,ytrain,ytest = train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
xtrain=xtrain/255.0
xtest = xtest/255.0

clf = LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrain,ytrain)

ypred = clf.predict(xtest)
accuracy = accuracy_score(ypred,ytest)
print(accuracy)


cap = cv2.VideoCapture(0)
while True:
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        upperLeft = (int(width/2 - 56),int(height/2 -56))
        bottomRight = (int(width/2 + 56),int(height/2 +56))
        cv2.rectangle(gray,upperLeft,bottomRight,(0,255,0),2)
        roi = gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]
        impil = Image.fromarray(roi)
        image_bw = impil.convert('L')
        image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixelfilter = 20
        minpixel = np.percentile(image_bw_resized_inverted,pixelfilter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted- minpixel,0,255)
        maxpixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/maxpixel
        testSample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        testpred = clf.predict(testSample)
        print("Number is", testpred)
        cv2.imshow('frame',gray)

        if cv2.waitKey(1) and 0xFF==ord('q'):
             break
    except Exception as e:
        pass


cap.release()
cv2.destroyAllWindows()