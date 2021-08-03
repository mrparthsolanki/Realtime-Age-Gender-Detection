# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:27:41 2020

@author: lenovo
"""
 
import cv2
import face_recognition

web_video_stream=cv2.VideoCapture(0)
all_face_locations=[]

while True: 
    ret,current_frame=web_video_stream.read()
    current_fram_small=cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    all_face_locations=face_recognition.face_locations(current_fram_small,model='hog')
    
    for index,current_face_location in enumerate(all_face_locations):
        top_pos,right_pos,bottom_pos,left_pos=current_face_location
        top_pos=top_pos*4
        right_pos=right_pos*4
        bottom_pos=bottom_pos*4
        left_pos=left_pos*4
        print('found face{} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos, right_pos,bottom_pos,left_pos))
 
        current_face_image=current_frame[top_pos:bottom_pos,left_pos:right_pos]
        
        Age_gender_model_values=(78.4263377603,87.7689143744,114.895847746)
        
        current_face_image_blob=cv2.dnn.blobFromImage(current_face_image,1,(227,227),Age_gender_model_values,swapRB=False)
        
        gender_label_list=['Male','Female']
        gender_protext="Dataset/gender_deploy.prototxt"
        gender_caffemodel="Dataset/gender_net.caffemodel"
        
        gender_cov_net=cv2.dnn.readNet(gender_caffemodel,gender_protext)
        
        gender_cov_net.setInput(current_face_image_blob)
        
        gender_predictions=gender_cov_net.forward()
        
        gender=gender_label_list[gender_predictions[0].argmax()]
        
        
        age_label_list=['(0-2)','(4-6)','(8-12)','(15-20)','(21-24)','(25-32)','(38-43)','(48-53)','(60-100)']
        
        age_protext="Dataset/age_deploy.prototxt"
        age_caffemodel="Dataset/age_net.caffemodel"
        
        age_cov_net=cv2.dnn.readNet(age_caffemodel,age_protext)
        
        age_cov_net.setInput(current_face_image_blob)
        
        age_predictions=age_cov_net.forward()
        
        age=age_label_list[age_predictions[0].argmax()]
        
        
        



        cv2.rectangle(current_frame,(left_pos,top_pos), (right_pos,bottom_pos), (0,0,255),2)
        
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, gender+" "+age+"yrs", (left_pos,bottom_pos+20), font, 0.5,(0,255,0),1)
        
        
    cv2.imshow("Webcam Video",current_frame)
        



    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

web_video_stream.release()
cv2.destroyAllWindows() 
