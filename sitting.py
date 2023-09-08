import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

import argparse
body_estimation = Body('model/body_pose_model.pth')

np.set_printoptions(suppress=True)
# This right??
#keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

ap = argparse.ArgumentParser()
ap.add_argument('img', help='Directory of raw frames')
args = ap.parse_args()



# Calculate the angle between the thighs (angle between hip and knee points)
def calculate_angle(point1, point2, point3):
    v1 = point1 - point2
    v2 = point3 - point2
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle_rad = np.arccos(dot_product / (norm_product + 1e-6))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


frame_limit = 20000
test_image = args.img
#while current_frame <= frame_limit:
oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)
print('body Candidate')
print(type(candidate))
print(candidate)
#print('body subset')
#print(subset)


# Define the indices of key points for hips and knees
left_hip_index = 11
left_knee_index = 12
right_hip_index = 8
right_knee_index = 9


for sub in subset:
    print(sub)

    # Extract the x and y coordinates of the relevant key points
    if sub[left_hip_index] == -1 or sub[left_knee_index] == -1 or sub[right_hip_index] == -1 or sub[right_knee_index] == -1:
        pose_classification = "Standing"
    else:
        left_hip = candidate[int(sub[left_hip_index])][:2]
        left_knee = candidate[int(sub[left_knee_index])][:2]
        right_hip = candidate[int(sub[right_hip_index])][:2]
        right_knee = candidate[int(sub[right_knee_index])][:2]



        left_leg_angle = abs(left_hip[1] - left_knee[1]) /  abs(left_hip[0] - left_knee[0])
        right_leg_angle = abs(right_hip[1] - right_knee[1]) / abs(right_hip[0] - right_knee[0])

#        # Calculate the angles for both legs
#        left_leg_angle = calculate_angle(left_hip, left_knee, right_knee)
#        right_leg_angle = calculate_angle(right_hip, right_knee, left_knee)
#
#        # Set a threshold to classify standing and sitting
#        standing_threshold = 70  # Adjust this threshold based on your data and model's accuracy
#
#        # Determine if the person is standing or sitting
        if left_leg_angle < 1.5 or right_leg_angle < 1.5:
            pose_classification = "Sitting"
        else:
            pose_classification = "Standing"

        print('left hip ', left_hip)
        print('left knee ', left_knee)
        print('left_leg_angle ', left_leg_angle)
        print('right hip ', right_hip)
        print('right knee ', right_knee)
        print('right_leg_angle ', right_leg_angle)

    left = 10000
    right = 0
    top = 10000
    bottom = 0 
    for idx in range(0,len(sub)):
        if candidate[int(sub[idx])][0] < left:
            left = candidate[int(sub[idx])][0]
        if candidate[int(sub[idx])][0] > right:
            right = candidate[int(sub[idx])][0]
        if candidate[int(sub[idx])][1] < top:
            top = candidate[int(sub[idx])][1]
        if candidate[int(sub[idx])][1] > bottom:
            bottom = candidate[int(sub[idx])][1]

    print("Pose Classification:", pose_classification)
    print()





