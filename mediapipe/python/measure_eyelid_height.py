import cv2
# import mediapipe as mp
import mediapipe.python as mp
import scipy as sp
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt

LEFT_EYE = [33,246,161,160,159,158,157,173,133,7,163,144,145,153,154,155] #159 - 145
RIGHT_EYE = [362,398,384,385,386,387,388,466,163,382,381,380,374,373,390,249] # 386 - 374
IMAGE_FILES = ['Jeny.jpeg']

def measure_eyelid_heights(IMAGE_FILES):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    # For static images:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    idx_to_coordinates = dict()
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        for idx, file in enumerate(IMAGE_FILES):
            print(idx, file)
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # results_left_eye = mp_left_eye(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            counter = 0
            for face_landmarks in results.multi_face_landmarks:
                # print(counter)
                # print('face_landmarks:', face_landmarks)

                # mp_drawing.draw_landmarks(
                #     image=annotated_image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_tesselation_style())

                # mp_drawing.draw_landmarks(
                #     image=annotated_image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_contours_style())
                idx_to_coordinates = mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    # connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                counter += 1
                # cv2.imwrite('./tmp/annotated_image' + str(idx) + '.png', annotated_image)
                # plt.imshow(annotated_image)

                coords = list()
                coords.append(idx_to_coordinates[159])
                coords.append(idx_to_coordinates[145])
                coords.append(idx_to_coordinates[386])
                coords.append(idx_to_coordinates[374])

                print("Left Eyelid height :",distance.cdist(coords, coords, 'euclidean')[0][1])
                print("Right Eyelid height :",distance.cdist(coords, coords, 'euclidean')[2][3])



measure_eyelid_heights(IMAGE_FILES)