import cv2
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        
        success, image = self.video.read()
        img_counter=0
        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        img_name = "opencv_frame_{}.jpg".format(img_counter)
        
        cv2.imwrite('/home/arko/Documents/Python_Scripts/Capstone_Project/open_cv/' + img_name, image)
        print("{} written!".format(img_name))
        img_counter+=1
                
        for (x,y,w,h) in face_rects:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()