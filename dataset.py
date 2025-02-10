import mediapipe as mp
from mediapipe.tasks import python
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pprint

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
)
handlandmarkoptions = HandLandmarker.create_from_options(options)
key_points = []


with handlandmarkoptions as landmarker:
    img = Image.open('asl_set/1/Input Images - Sign 1/Sign 1 (1).jpeg')
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))
    landmarks = landmarker.detect(img)

    pprint.pprint(landmarks)

    for landmark in landmarks.hand_world_landmarks[0]:
        key_points.append([landmark.x, landmark.y, landmark.z])

key_points = np.array(key_points)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')



handkeypoints = [[0, 1, 2, 3, 4], # thumb
                 [0, 5, 6, 7, 8], #index
                 [5, 9, 10, 11, 12], # middle
                 [9, 13, 14, 15, 16], # ring
                 [13, 17, 18, 19, 20], # pinky
                 [0, 17], #wrist to pinky? 
]
#plotting the hand keypoints and drawing them with lines
for indices in handkeypoints:
    xs = [key_points[i, 0] for i in indices]
    ys = [key_points[i, 1] for i in indices]
    zs = [key_points[i, 2] for i in indices]
    plt.plot(xs, ys, zs)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
