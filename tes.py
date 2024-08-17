import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(1)
detector = FaceMeshDetector()

ids = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

while True:
    success, img = cap.read()
    
    img, faces = detector.findFaceMesh(img, draw=False)
    
    if faces:
        face = faces[0]
        for id in ids:
            cv2.circle(img,face[id],5,(0,0,255),cv2.FILLED)

            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            leftLenghtVer, _=  detector.findDistance(leftUp,leftDown)
            
            rightUp = face[386]
            rightDown = face[374]
            rightLeft = face[382]
            rightRight = face[263]
            rightLenghtVer, _=  detector.findDistance(rightUp,rightDown)

            cv2.line(img,leftUp,leftDown,(0,200,0), 3)
            
            cv2.line(img,rightUp,rightDown,(0,200,0), 3)

            # for id, point in enumerate(face):
            #     if id in ids:
            #         cv2.putText(img, str(id), point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 