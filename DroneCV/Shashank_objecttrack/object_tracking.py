
import cv2
from imutils.video import FPS
tracker=cv2.TrackerCSRT_create()
#initialize the bounding box
initbb=None
fps=None

cap=cv2.VideoCapture(0)
while True:
    _,frame=cap.read()
    #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame=cv2.resize(frame,(640, 480))
    (H,W) =frame.shape[:2]
    if initbb is not None:
        (sucess,box)=tracker.update(frame)
        if sucess:
            (x,y,w,h)=[int(v) for v in box]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        fps.update()
        fps.stop()
        info=[("Tracker","CSRT"),
               ("Success", "yes" if sucess else "No"),
               ("FPS", "{:.2f}".format(fps.fps())),]
        		# loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text="{}:{}".format(k,v)
            cv2.putText(frame,text,(10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key==ord("s"):
        initbb=cv2.selectROI("Frame",frame,fromCenter=False,showCrosshair=True)
        tracker.init(frame,initbb)
        fps = FPS().start()
    
    elif key==ord('q'):
        break

cv2.destroyAllWindows()
