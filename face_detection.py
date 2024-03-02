import cv2 as cv
img=cv.imread(r'C:\Users\Somashekar\OneDrive\Desktop\ML-PRACTISE\face_dataset\Smile\smile1.jpg')
cv.imshow('taken image',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('grayscale image',gray)
# blur=cv.GaussianBlur(gray,(3,3),cv.BORDER_DEFAULT)
# cv.imshow('blurred',blur)
haar_c=cv.CascadeClassifier(r'C:\Users\Somashekar\OneDrive\Desktop\ML-PRACTISE\haarcascade_frontface_default.xml')
rect_list=haar_c.detectMultiScale(gray,1.1,6)
#drawing detected face rectangles
for (x,y,w,h) in rect_list:
    cv.rectangle(img,(x,y),(x+w,y+h),thickness=2,color=(0,255,0))
cv.imshow('faces detected',img)
print(f'NUMBER OF FACES DETECTED ARE: {len(rect_list)}')
cv.waitKey(0)