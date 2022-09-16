import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime

path='imagesBasic'
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)

def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markattendence(name):
    with open('information.csv','r+') as f:
        mydatalist = f.readlines()
        namelist=[]
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')




encodelistknown = findencodings(images)
print('encoding complete')
cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgs)
    encodesCurFrame = face_recognition.face_encodings(imgs, facesCurFrame)


    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodelistknown,encodeFace)
        facedis = face_recognition.face_distance(encodelistknown,encodeFace)
        #print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
           # print(name)
            y1,x2,y2,x1 = faceLoc
            now1 = datetime.now()
            dtstring1 = now1.strftime('%H:%M:%S')
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name+dtstring1,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendence(name)
        else:
            y1,x2,y2,x1 = faceLoc
            now2 = datetime.now()
            dtstring2=now2.strftime('%H:%M:%S')
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,'unknown'+dtstring2,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            markattendence('unknown')
            count=0
            path='imagesBasic'
            images1 = []
            classnames1 = []
            mylist1 = os.listdir(path)
            print(mylist1)
            for cl in mylist1:
             curimg1 = cv2.imread(f'{path}/{cl}')
             images1.append(curimg1)
             classnames1.append(os.path.splitext(cl)[0])
            print(classnames1)
            encodelistknown1 = findencodings(images1)
            # cam = cv2.VideoCapture(0)
            now1 = datetime.now()
            pt=int(now1.strftime('%S'))
            ut=pt+5;
            while pt<ut:
             pt+=1
             ret, imgimg=cap.read()
             imgs1=cv2.resize(imgimg,(0,0),None,0.25,0.25)
             imgs1=cv2.cvtColor(imgimg,cv2.COLOR_BGR2RGB)
             facesCurFrame1 = face_recognition.face_locations(imgs1)
             encodesCurframe1 = face_recognition.face_encodings(imgs1,facesCurFrame1)

             for encodeFace1,faceLoc1 in zip(encodesCurframe1,facesCurFrame1):
                matches1=face_recognition.compare_faces(encodelistknown1,encodeFace1)
                facedis1=face_recognition.face_distance(encodelistknown1,encodeFace1)
                matchindex1 = np.argmin(facedis1)

                if matches1[matchindex1]:
                    name1=classnames[matchindex1].upper()
                    now3 = datetime.now()
                    dtstring3=now3.strftime('%H:%M:%S')
                    y1,x2,y2,x1 = faceLoc1
                    cv2.rectangle(imgimg,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(imgimg,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(imgimg,name+dtstring3,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                    markattendence(name)
                else:
                    y1,x2,y2,x1 = faceLoc1
                    now4 = datetime.now()
                    dtstring4=now4.strftime('%H:%M:%S')
                    cv2.rectangle(imgimg,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.rectangle(imgimg,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
                    cv2.putText(imgimg,'unknown'+dtstring4,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                    markattendence('unknown')

             cv2.imshow('test', imgimg)
             

             if not ret:
               break

             k=cv2.waitKey(1)

            #  if k%256==27:
            #   print("close")
            #   break
            #  elif k%256==32:
             file = 'C:/Users/suhas/OneDrive/Desktop/snapshot/imgimg'+str(count)+'.jpg'
             cv2.imwrite(file,imgimg)
             count+=1
            cv2.destroyWindow('test')
            
    cv2.imshow('webcam',img)
    cv2.waitKey(1)