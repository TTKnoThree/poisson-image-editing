import face_recognition
import cv2
import os
import numpy as np

filename = '5_source'
image = face_recognition.load_image_file("input_images\\{}.png".format(filename))
face_locations = face_recognition.face_locations(image)
face_landmarks = face_recognition.face_landmarks(image)
rectangle = []
points = []
gray_bound = []
white_bound = []
dilate_scale = 1.8

for face in face_locations:
    # print(face) # top, right, bottom, left
    h_mid = round((face[0] + face[2]) / 2)
    w_mid = round((face[1] + face[3]) / 2)
    h_dilate = round((face[2] - face[0]) * dilate_scale)
    w_dilate = round((face[1] - face[3]) * dilate_scale)
    point1 = (max(round(w_mid - w_dilate/2),0), max(round(h_mid - h_dilate/2),0))
    point2 = (min(round(w_mid + w_dilate/2),image.shape[1]), min(round(h_mid + h_dilate/2),image.shape[1]))
    rectangle.append((point1, point2))

for face in face_landmarks:
    points_dit = {}
    for key in ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye']:
        # 'right_eyebrow','left_eyebrow': 5 points
        # 'right_eye','left_eye': 6 points
        value = face[key]
        xs = [i[0] for i in value]
        ys = [i[1] for i in value]
        x_mean = np.array(xs).sum()/len(xs)
        y_mean = np.array(ys).sum()/len(ys)
        points_dit[key] = (x_mean, y_mean)
    points.append(points_dit)

for i in range(len(points)):
    point_dict = points[i]
    white_top = 3*point_dict['left_eye'][1] - 2*point_dict['left_eyebrow'][1]
    white_bottom = min(4*point_dict['left_eye'][1] - 3*point_dict['left_eyebrow'][1],image.shape[0])
    white_left = point_dict['left_eye'][0]
    white_right = point_dict['right_eye'][0]
    white_bound.append((int(white_left), int(white_right), int(white_top), int(white_bottom)))

for i in range(len(points)):
    point_dict = points[i]
    gray_left = max(2*point_dict['left_eyebrow'][0] - point_dict['right_eyebrow'][0],0)
    gray_right = min(2*point_dict['right_eyebrow'][0] - point_dict['left_eyebrow'][0],image.shape[1])
    gray_top = (point_dict['left_eyebrow'][1]+point_dict['right_eyebrow'][1])//2
    gray_bottom = rectangle[i][1][1] # bottom of face recognition
    gray_bound.append((int(gray_left), int(gray_right), int(gray_top), int(gray_bottom)))

for rect in rectangle:
    cv2.rectangle(image,rect[0],rect[1],(0,255,0),2)

trimap = np.zeros(image.shape, dtype=np.uint8)
for gray in gray_bound:
    trimap[gray[2]:gray[3],gray[0]:gray[1],:] = 128
for white in white_bound:
    trimap[white[2]:white[3],white[0]:white[1],:] = 255

os.makedirs("face_rec", exist_ok=True)
cv2.imwrite("face_rec\\"+filename+'.jpg',image[:,:,(2,1,0)])
trimap_name = "face_rec\\"+filename.split('_')[0]+'_trimap.jpg'
cv2.imwrite(trimap_name,trimap)