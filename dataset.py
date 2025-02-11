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

options = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
))

key_points = []

with options as landmarker:
    img = Image.open('asl_set/1/Input Images - Sign 1/Sign 1 (1).jpeg')
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))
    landmarks = landmarker.detect(img)

    pprint.pprint(landmarks)

    for landmark in landmarks.hand_world_landmarks[0]:
        key_points.append([landmark.x, landmark.y, landmark.z])

key_points = np.array(key_points)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

hand_landmark_indices = [
    [0, 5, 9, 13, 17, 0],  # Palm
    [0, 1, 2, 3, 4],  # Thumb
    [5, 6, 7, 8],  # Index
    [9, 10, 11, 12],  # Middle
    [13, 14, 15, 16],  # Ring
    [17, 18, 19, 20],  # Pinky
]

# Plotting the hand landmarks and drawing them with lines
for indices in hand_landmark_indices:
    xs = [key_points[i, 0] for i in indices]
    ys = [key_points[i, 1] for i in indices]
    zs = [key_points[i, 2] for i in indices]

    plt.plot(xs, ys, zs)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
