import cv2
import numpy as np
import time

from keras.models import load_model
model = load_model("keras_model_1.h5", compile=False)
model_a=load_model("keras_model.h5", compile=False)

v=cv2.VideoCapture(0)

f_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
e_cas=cv2.CascadeClassifier('haarcascade_eye.xml')
m_cas=cv2.CascadeClassifier('mouth.xml')


font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:    
    flag,frame=v.read()
    if flag:

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
       
            
        face=f_cas.detectMultiScale(frame,1.3,5)
        
        for(x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            roi_a = gray[y:y+h, x:x+w]
            roi_b = frame[y:y+h, x:x+w]
            
            eyes=e_cas.detectMultiScale(roi_a)                                      
            for (ex,ey,ew,eh) in eyes:
                image = cv2.resize(roi_b, (224, 224), interpolation=cv2.INTER_AREA)
                image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
                image = (image / 127.5) - 1
                # Predicts the model
                prediction_a = model.predict(image)
                index = np.argmax(prediction_a)                
                confidence_score_a = prediction_a[0][index]
                
                print("Confidence Score:", str(np.round(confidence_score_a * 100))[:-2], "%")
                print('Pred Score:',confidence_score_a)
                cv2.putText(frame,'Pred Score(eyes):'+str(confidence_score_a),(10,20), font, 1,(255,255,255),1,cv2.LINE_AA)

                if(confidence_score_a)<=0.8:
                    cv2.rectangle(roi_b,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                    cv2.putText(frame,"EYES:- close",(10,40), font, 1,(255,0,0),1,cv2.LINE_AA)
                    html_string = """
                            <audio controls autoplay>
                            <source src="https://www.orangefreesounds.com/wp-content/uploads/2023/09/80s-alarm-clock-sound.mp3" type="audio/mp3">
                            </audio>
                            """

                            sound = st.empty()
                            sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                            time.sleep(3)  # wait for 2 seconds to finish the playing of the audio
                            sound.empty()  # optionally delete the element afterwards
                            
                else:
                    cv2.rectangle(roi_b,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
                    cv2.putText(frame,"EYES:- open",(10,40), font, 1,(0,0,255),1,cv2.LINE_AA)
                                
            mouth=m_cas.detectMultiScale(frame,1.7,11)
            for(x,y,w,h) in mouth:
                y = int(y - 0.15*h)
                img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
                img = (img / 127.5) - 1
                # Predicts the model
                prediction_b = model_a.predict(img)
                index = np.argmax(prediction_b)                
                confidence_score_b = prediction_b[0][index]

                print("Confidence Score (mouth):", str(np.round(confidence_score_b * 100))[:-2], "%")
                print('Pred Score(mouth):',confidence_score_b)
                cv2.putText(frame,'Pred Score(mouth):'+str(confidence_score_b),(10,70), font, 1,(255,255,255),1,cv2.LINE_AA)

                if(confidence_score_b)>=0.8:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)                
                    cv2.putText(frame,"MOUTH:- open",(10,90), font, 1,(0,0,255),1,cv2.LINE_AA)
                    html_string = """
                            <audio controls autoplay>
                            <source src="https://www.orangefreesounds.com/wp-content/uploads/2023/09/80s-alarm-clock-sound.mp3" type="audio/mp3">
                            </audio>
                            """

                            sound = st.empty()
                            sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                            time.sleep(3)  # wait for 2 seconds to finish the playing of the audio
                            sound.empty()  # optionally delete the element afterwards
                            
                else:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)                
                    cv2.putText(frame,"MOUTH:- close",(10,90), font, 1,(0,255,0),1,cv2.LINE_AA)                                                                            
                
        cv2.namedWindow('W',cv2.WINDOW_NORMAL)
        cv2.imshow('W',frame)
        k=cv2.waitKey(30)
        if(k==ord('a')):
            break
    else:
        break
v.release()
cv2.destroyAllWindows()


w=cv2.VideoCapture('http://192.168.0.105:8080/video')

f_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
e_cas=cv2.CascadeClassifier('haarcascade_eye.xml')
m_cas=cv2.CascadeClassifier('mouth.xml')


font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:    
    flag,frame=w.read()
    if flag:

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
       
            
        face=f_cas.detectMultiScale(frame,1.3,5)
        
        for(x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            roi_a = gray[y:y+h, x:x+w]
            roi_b = frame[y:y+h, x:x+w]
            
            eyes=e_cas.detectMultiScale(roi_a)                                      
            for (ex,ey,ew,eh) in eyes:
                image = cv2.resize(roi_b, (224, 224), interpolation=cv2.INTER_AREA)
                image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
                image = (image / 127.5) - 1
                # Predicts the model
                prediction_a = model.predict(image)
                index = np.argmax(prediction_a)                
                confidence_score_a = prediction_a[0][index]
                
                print("Confidence Score:", str(np.round(confidence_score_a * 100))[:-2], "%")
                print('Pred Score:',confidence_score_a)
                cv2.putText(frame,'Pred Score(eyes):'+str(confidence_score_a),(10,20), font, 1,(255,255,255),1,cv2.LINE_AA)

                if(confidence_score_a)<=0.8:
                    cv2.rectangle(roi_b,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                    cv2.putText(frame,"EYES:- close",(10,40), font, 1,(255,0,0),1,cv2.LINE_AA)
                    html_string = """
                            <audio controls autoplay>
                            <source src="https://www.orangefreesounds.com/wp-content/uploads/2023/09/80s-alarm-clock-sound.mp3" type="audio/mp3">
                            </audio>
                            """

                            sound = st.empty()
                            sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                            time.sleep(3)  # wait for 2 seconds to finish the playing of the audio
                            sound.empty()  # optionally delete the element afterwards
                            
                else:
                    cv2.rectangle(roi_b,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
                    cv2.putText(frame,"EYES:- open",(10,40), font, 1,(0,0,255),1,cv2.LINE_AA)
                                
            mouth=m_cas.detectMultiScale(frame,1.7,11)
            for(x,y,w,h) in mouth:
                y = int(y - 0.15*h)
                img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
                img = (img / 127.5) - 1
                # Predicts the model
                prediction_b = model_a.predict(img)
                index = np.argmax(prediction_b)                
                confidence_score_b = prediction_b[0][index]

                print("Confidence Score (mouth):", str(np.round(confidence_score_b * 100))[:-2], "%")
                print('Pred Score(mouth):',confidence_score_b)
                cv2.putText(frame,'Pred Score(mouth):'+str(confidence_score_b),(10,70), font, 1,(255,255,255),1,cv2.LINE_AA)

                if(confidence_score_b)>=0.8:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)                
                    cv2.putText(frame,"MOUTH:- open",(10,90), font, 1,(0,0,255),1,cv2.LINE_AA)
                    html_string = """
                            <audio controls autoplay>
                            <source src="https://www.orangefreesounds.com/wp-content/uploads/2023/09/80s-alarm-clock-sound.mp3" type="audio/mp3">
                            </audio>
                            """

                            sound = st.empty()
                            sound.markdown(html_string, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
                            time.sleep(3)  # wait for 2 seconds to finish the playing of the audio
                            sound.empty()  # optionally delete the element afterwards
                            
                else:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)                
                    cv2.putText(frame,"MOUTH:- close",(10,90), font, 1,(0,255,0),1,cv2.LINE_AA)                                                                            
                
        cv2.namedWindow('W',cv2.WINDOW_NORMAL)
        cv2.imshow('W',frame)
        k=cv2.waitKey(30)
        if(k==ord('a')):
            break
    else:
        break
w.release()
cv2.destroyAllWindows()
