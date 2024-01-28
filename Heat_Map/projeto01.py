import cv2
import numpy as np
from ultralytics import YOLO

video = cv2.VideoCapture("seu_video.mp4")

modelo = YOLO('yolov8x.pt')

blankImage = np.ones([720,1270],np.uint32)

while True:

    check,img = video.read()
    img = cv2.resize(img,(1270,720))

    objetos = modelo(img,stream=True)

    for objeto in objetos:
        info = objeto.boxes
        for box in info:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            conf = int(box.conf[0]*100)/100
            cls = int(box.cls[0])
            if cls == 0:
                blankImage[y1:y2,x1:x2] += 1

        blankImageNorm = 255*((blankImage-blankImage.min()))/(blankImage.max()-blankImage.min())

        blankImageNorm = blankImageNorm.astype('uint8')
        blankImageNorm = cv2.GaussianBlur(blankImageNorm,(9,9),0)
        heatMap = cv2.applyColorMap(blankImageNorm,cv2.COLORMAP_JET)
        imgFinal = cv2.addWeighted(heatMap,0.5,img,0.5,0)


            #cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),4)



    cv2.imshow('Img',img)
    cv2.imshow('Blanck',imgFinal)
    cv2.waitKey(1)