import numpy as np
import dlib
import cv2
import math as m

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

'''duz = cv2.imread('assets/duz.jpg')
duz_2 = cv2.imread('assets/duz_2.jpg')
orta_alt = cv2.imread('assets/orta_alt.jpg')
orta_alt_2 = cv2.imread('assets/orta_alt_2.jpg')
orta_ust = cv2.imread('assets/orta_ust.jpg')
sag_alt = cv2.imread('assets/sag_alt.jpg')
sag_ust = cv2.imread('assets/sag_ust.jpg')
sol_alt = cv2.imread('assets/sol_alt.jpg')
sol_ust = cv2.imread('assets/sol_ust.jpg')

images = [duz, duz_2, orta_alt, orta_alt_2, orta_ust, sag_alt, sag_ust, sol_alt, sol_ust]
images_resized = []
images_flip = []

for img in images: #çevirdik griye aktardık.
    img = cv2.flip(img,180)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Gerisi silinebilir.
    images_flip.append(img)
    img_resized = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    images_resized.append(img_resized)'''






cap = cv2.VideoCapture(0)

detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#for img in images_resized:
while True:

    #frame = img
    ret,frame = cap.read()
    frame = cv2.flip(frame,180)
    canvas = cv2.imread('assets/canvas.jpeg')
    canvas = cv2.resize(canvas,(0,0), fx = 0.3, fy= 0.3)
    #frame = cv2.imread('assets/foto_5.jpg')
    #frame = cv2.resize(frame,(0,0), fx= 0.3, fy = 0.3)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect(gray_frame)

    for face in faces :# gerekli değil silinecek. Demo için duruyor.
        face_Low_X, face_High_X, face_Low_Y, face_High_Y= face.left(), face.right(), face.top(), face.bottom()
        cv2.rectangle(frame, (face_Low_X, face_Low_Y), (face_High_X, face_High_Y),(0,225,0),2)
        landmarks = predictor(frame, face)
        print(landmarks)

        nokta_36 = (landmarks.part(36).x,landmarks.part(36).y)
        nokta_37 = (landmarks.part(37).x, landmarks.part(37).y)
        nokta_38 = (landmarks.part(38).x, landmarks.part(38).y)
        nokta_39 = (landmarks.part(39).x, landmarks.part(39).y)
        nokta_40 = (landmarks.part(40).x, landmarks.part(40).y)
        nokta_41 = (landmarks.part(41).x, landmarks.part(41).y)
        nokta_42 = (landmarks.part(42).x, landmarks.part(42).y)
        nokta_43 = (landmarks.part(43).x, landmarks.part(43).y)
        nokta_44 = (landmarks.part(44).x, landmarks.part(44).y)
        nokta_45 = (landmarks.part(45).x, landmarks.part(45).y)
        nokta_46 = (landmarks.part(46).x, landmarks.part(46).y)
        nokta_47 = (landmarks.part(47).x, landmarks.part(47).y)

        left_eye_location = np.array([nokta_36, nokta_37, nokta_38, nokta_39, nokta_40, nokta_41], np.int32)
        right_eye_location = np.array([nokta_42, nokta_43, nokta_44, nokta_45, nokta_46, nokta_47], np.int32)

        left_eye_min_X = np.min(left_eye_location[:, 0])
        left_eye_max_X = np.max(left_eye_location[:, 0])
        left_eye_min_Y = np.min(left_eye_location[:, 1])
        left_eye_max_Y = np.max(left_eye_location[:, 1])

        right_eye_min_X = np.min(right_eye_location[:, 0])
        right_eye_max_X = np.max(right_eye_location[:, 0])
        right_eye_min_Y = np.min(right_eye_location[:, 1])
        right_eye_max_Y = np.max(right_eye_location[:, 1])

        frame_gauss = cv2.GaussianBlur(frame, (13, 13), 10) #original value 11,11,10

        gray_frame_gauss = cv2.cvtColor(frame_gauss,cv2.COLOR_BGR2GRAY)


        frame_eroded = cv2.erode(gray_frame_gauss, None, iterations=2) # org val 2

        frame_dilated = cv2.dilate(frame_eroded, None, iterations=4) # org val 4


        '''blank = np.zeros(frame.shape[:2], dtype='uint8')
    
        mask = cv2.fillPoly(blank, [left_eye_location], 255)
        masked = cv2.bitwise_and(frame_dilated, frame_dilated, mask=mask)
    
        mask = cv2.fillPoly(blank, [right_eye_location], 255)
        masked = cv2.bitwise_and(frame_dilated, frame_dilated, mask=mask)'''

        #masked = frame_dilated

        _, frame_thres = cv2.threshold(frame_dilated, 105 , 255, cv2.THRESH_BINARY)#fotoğrafta 75 te çalışıyor.

        left_eye_forged = frame_thres[left_eye_min_Y:left_eye_max_Y, left_eye_min_X:left_eye_max_X]
        left_eye_forged = cv2.resize(left_eye_forged, (0, 0), fx=10, fy=10)

        right_eye_forged = frame_thres[right_eye_min_Y:right_eye_max_Y, right_eye_min_X:right_eye_max_X]
        right_eye_forged = cv2.resize(right_eye_forged, (0, 0), fx=10, fy=10)

        left_eye = frame[left_eye_min_Y:left_eye_max_Y, left_eye_min_X:left_eye_max_X]
        left_eye = cv2.resize(left_eye, (0, 0), fx=10, fy=10)

        right_eye = frame[right_eye_min_Y:right_eye_max_Y, right_eye_min_X:right_eye_max_X]
        right_eye = cv2.resize(right_eye, (0, 0), fx=10, fy=10)

        left_eye_height = left_eye.shape[0]
        left_eye_width = left_eye.shape[1]

        canvas_height = canvas.shape[0]
        canvas_width = canvas.shape[1]

        right_eye_height = right_eye.shape[0]
        right_eye_width = right_eye.shape[1]

        section_set_height, section_set_width = 10, 20

        cursor_left_height = int(section_set_height/2)
        cursor_left_width = int(section_set_width/2)

        cursor_right_height = int(section_set_height / 2)
        cursor_right_width = int(section_set_width / 2)

        left_eye_height_multipler = int(left_eye_height/section_set_height)
        left_eye_width_multipler = int(left_eye_width/section_set_width)

        canvas_height_multipler = int(canvas_height/section_set_height)
        canvas_width_multipler = int(canvas_width/section_set_width)

        right_eye_height_multipler = int(right_eye_height / section_set_height)
        right_eye_width_multipler = int(right_eye_width / section_set_width)

        references_drawing_left = [[0 for h in range(section_set_height+1)] for w in range(section_set_width+1)] #references_drawing[height,width]
        references_forging_left = [[0 for h in range(section_set_height+1)] for w in range(section_set_width+1)] #references_forging[height,width]

        references_drawing_canvas = [[0 for h in range(section_set_height+1)] for w in range(section_set_width+1)] #references_drawing[height,width]
        references_forging_canvas = [[0 for h in range(section_set_height+1)] for w in range(section_set_width+1)] #references_forging[height,width]

        references_drawing_right = [[0 for h in range(section_set_height+1)] for w in range(section_set_width+1)]  # references_drawing[height,width]
        references_forging_right = [[0 for h in range(section_set_height+1)] for w in range(section_set_width+1)]  # references_forging[height,width]


        for height in range (section_set_height+1):
            for width in range (section_set_width+1):

                w_left = width * left_eye_width_multipler
                h_left = height * left_eye_height_multipler

                if height == section_set_height:
                    h_left = h_left-5
                if width == section_set_width:
                    w_left = w_left-5

                coordinates = (h_left, w_left)
                references_forging_left[width][height] = coordinates

                coordinates = (w_left, h_left)
                references_drawing_left[width][height] = coordinates


                w_right = width * right_eye_width_multipler
                h_right = height * right_eye_height_multipler

                if height == section_set_height:
                    h_right = h_right-5
                if width == section_set_width:
                    w_right = w_right-5

                coordinates = (h_right, w_right)
                references_forging_right[width][height] = coordinates

                coordinates = (w_right, h_right)
                references_drawing_right[width][height] = coordinates


                w_canvas = width * canvas_width_multipler
                h_canvas = height * canvas_height_multipler

                if height == section_set_height:
                    h_left = h_left-5
                if width == section_set_width:
                    w_left = w_left-5

                coordinates = (h_canvas, w_canvas)
                references_forging_canvas[width][height] = coordinates

                coordinates = (w_canvas, h_canvas)
                references_drawing_canvas[width][height] = coordinates


        for h in range (section_set_height+1):
            for w in range (section_set_width+1):


                if not (left_eye_forged[references_forging_left[w][h]]):

                    if w < cursor_left_width:
                        cursor_left_width = cursor_left_width -2
                    if w > cursor_left_width:
                        cursor_left_width = cursor_left_width +2
                    if h < cursor_left_height:
                        cursor_left_height = cursor_left_height -1
                    if h > cursor_left_height:
                        cursor_left_height = cursor_left_height +1

                    if w < cursor_right_width:
                        cursor_right_width = cursor_right_width -2
                    if w > cursor_right_width:
                        cursor_right_width = cursor_right_width +2
                    if h < cursor_right_height:
                        cursor_right_height = cursor_right_height -1
                    if h > cursor_right_height:
                        cursor_right_height = cursor_right_height +1

                cv2.circle(left_eye,references_drawing_left[w][h],2,blue,2)
                cv2.circle(right_eye, references_drawing_right[w][h],2,blue,2)
                cv2.circle(canvas, references_drawing_canvas[w][h],2,blue,4)
                print(cursor_left_width, cursor_left_height, cursor_right_height, cursor_right_width)

        canvas = cv2.circle(canvas,(references_drawing_canvas[cursor_left_width][cursor_left_height-2]),25,blue,10)        
        canvas = cv2.circle(canvas, (references_drawing_canvas[cursor_right_width][cursor_right_height-2]), 25, green,10)

        cv2.circle(left_eye,(references_drawing_left[cursor_left_width][cursor_left_height-2]),15,red,5)
        cv2.circle(right_eye, (references_drawing_right[cursor_right_width][cursor_right_height-2]), 15, red, 5)

        cv2.resize(frame,(0,0), fx= 0.5, fy = 0.5)
        cv2.imshow('frame',frame)
        #cv2.imwrite('Ana_Goruntu.jpg',frame)
        
        cv2.imshow('canvas',canvas)
        

        cv2.resize(frame_gauss,(0,0), fx= 0.7, fy = 0.7)
        cv2.imshow('frame gauss', frame_gauss)
        #cv2.imwrite('Ana_Goruntu_Gauss.jpg',frame_gauss)

        cv2.imshow('frame gauss grayed', gray_frame_gauss)
        #cv2.imwrite('Gaus_grayed.jpg',gray_frame_gauss)

        cv2.imshow('frame eroded', frame_eroded)
        #cv2.imwrite('Ana_Goruntu_Eroded.jpg',frame_eroded)

        cv2.imshow('frame dilated', frame_dilated)
        #cv2.imwrite('Ana_Goruntu_dilated.jpg',frame_dilated)

        cv2.imshow('frame thres', frame_thres)
        #cv2.imwrite('Thresholded.jpg',frame_thres)

        cv2.imshow('left eye', left_eye)
        #cv2.imwrite('left_eye_1.jpg',left_eye)
        
        cv2.imshow('right eye', right_eye)
        #cv2.imwrite('right_eye_1.jpg',right_eye)

        cv2.imshow('left eye forged', left_eye_forged)
        #cv2.imwrite('left_eye_forged_2.jpg',left_eye_forged)

        cv2.imshow('right eye forged', right_eye_forged)
        #cv2.imwrite('right_eye_forged_2.jpg',right_eye_forged)

    if cv2.waitKey(15) == ord('s'):
        break


cap.release()
cv2.destroyAllWindows()
