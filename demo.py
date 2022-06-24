import numpy as np
import cv2 as cv
from tensorflow import keras



# cap = cv.VideoCapture('D:/Machine_Learning/FINALL/video_afternoon/video_afternoon.mp4')

cap = cv.VideoCapture('C:/Users/thanh/OneDrive/Desktop/Project Image Processing/Traffic/FINALL/video_morning')

model = keras.models.load_model('C:/Users/thanh/Models/mymodel_o13')

#kernel_circle
kernel_ci = np.array([[0,0,1,0,0],
                      [0,1,1,1,0],
                      [1,1,1,1,1],
                      [0,1,1,1,0],
                      [0,0,1,0,0]], dtype=np.uint8)

# hsv image
def returnHSV(img):
    blur = cv.GaussianBlur(img,(5,5),0)
    hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
    return hsv


#Binary the image from hsv range
def binaryImg(img):

      image1 = img.copy()
      image2 = img.copy()
      image_blue = img.copy()

      hsv1 = returnHSV(image1)
      hsv2 = returnHSV(image2)
      hsvblue = returnHSV(image_blue)

      b_img1 = cv.inRange(hsv1,low_thresh1,high_thresh1)
      b_img2 = cv.inRange(hsv2,low_thresh2,high_thresh2)

      #binarize red sign image
      b_img_red = cv.bitwise_or(b_img1,b_img2)

      #binarize blue sign image
      b_img_blue = cv.inRange(hsvblue,low_thresh3,high_thresh3)
    #   b_img = cv.bitwise_or(b_img,b_img3)
      return b_img_red, b_img_blue


def findContour(img):
	contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	return contours


def boundaryBox(img,contours):
	box = cv.boundingRect(contours)
	sign = img[box[1]:(box[1]+box[3]) , box[0]:(box[0]+box[2])]
	return img, sign ,box


# preprocessing image
def preprocessingImageToClassifier(image=None,imageSize=48,mu=102.23982103497072,std= 72.11947698025735):
    image = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    image = cv.resize(image,(imageSize,imageSize))
    image = (image - mu) / std
    image = image.reshape(1,imageSize,imageSize,1)
    return image

# predict
def predict(sign):
	img = preprocessingImageToClassifier(sign,imageSize=48)
	return np.argmax(model.predict(img))

# finding the red sign
def findRedSign(frame):
    b_img_red, _ = binaryImg(frame)
    contours = findContour(b_img_red)
    for c in contours:
        area = cv.contourArea(c)
        if(area>1500):
            (a,b),r = cv.minEnclosingCircle(c)

            # checking the round shape or triangle shape of red sign
            if((area>0.42*np.pi*r*r)):
                img, sign, box = boundaryBox(frame,c)
                x,y,w,h = box

                 # checking the distance of top and bottom, aspect ratio of triangle and round shape
                if((w/h>0.7) and (w/h<1.2) and ((y+h)<0.6*height) and y>height/20):
                    label=labelToText[predict(sign)]
                    box = np.asarray(box)
                    rois.append(box)
                    labels.append(label)

# finding the blue sign
def findBlueSign(frame):
  _, b_img_blue = binaryImg(frame)
  contours_blue = findContour(b_img_blue)
  for c_blue in contours_blue:
        area_blue = cv.contourArea(c_blue)
        if(area_blue>1200):
            (a,b),r = cv.minEnclosingCircle(c_blue)
            area_circle = np.pi*r*r

            # checking the round shape of blue sign
            if(area_blue>0.7*area_circle):
              _, sign, box = boundaryBox(frame,c_blue)
              x,y,w,h = box

              # checking the distance of top and bottom; aspect ratio
              if((w/h>0.77) and (w/h<1.2)and(y+h)<0.6*height):
                label=labelToText[predict(sign)]
                box = np.asarray(box)
                rois.append(box)
                labels.append(label)
                
# label
# labelToText = { 0:"Stop",
#     			1:"Do not Enter",
#     			2:"No Parking",
#     			3:"Yeild",
# 				4:"Forbiden Road",  }

labelToText = { 0:"Speed 50",
    			1:"Yeild",
    			2:"Forbiden Road",
    			3:"Do not enter",
				4:"Walking",
				5:"Blue Straight",
                6:"Blue Straight left",
                7:"Blue Right",
                8:"Blue Circle", }


# Red
low_thresh1 = (165,100,40)
high_thresh1 = (179,255,255)

low_thresh2 = (0,160,40)
high_thresh2 = (10,255,255)

# Blue
low_thresh3 = (100,150,40)
high_thresh3 = (130,255,255)


isTracking = 0
frame_count = 0
max_trackingFrame = 10

while(cap.isOpened()):
    ret, frame = cap.read()

    height = frame.shape[0]
    width = frame.shape[1]

    if not ret:
            print(' can not read video frame. Video ended?')
            break

    # your code
    if isTracking == 0:
        # run detection code
        rois=[]
        labels=[]
        findRedSign(frame)
        findBlueSign(frame)
        # re-create and initilize the tracker
        trackers = cv.legacy.MultiTracker_create()
        for roi in rois:
            trackers.add(cv.legacy.TrackerCSRT_create(), frame, roi)
        isTracking = 1
    else: 
        if frame_count == max_trackingFrame:
            isTracking = 0
            frame_count = 0
        # update object location
        ret, objs = trackers.update(frame)
        if ret:
            label_count=0
            for obj in objs:

                # draw the bounding box and name of the traffic sign
                print(type(obj))
                p1 = (int(obj[0]), int(obj[1]))
                p2 = (int(obj[0]+obj[2]), int(obj[1]+obj[3]))
                cv.rectangle(frame, p1, p2, (0,255,0), 2)
                cv.rectangle(frame, p1, (int(obj[0]+2*obj[2]),int(obj[1]-15)), (0,255,0), -1)
                cv.putText(frame, labels[label_count], (int(obj[0]+(obj[2]/2)-5), int(obj[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                label_count = label_count+1
        else:
            print("tracking fail")
            isTracking = 0
        frame_count = frame_count + 1
    print('rois=',rois)
    cv.imshow('video', frame)
    if cv.waitKey(10) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
