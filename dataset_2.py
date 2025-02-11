import mediapipe as mp
from mediapipe.tasks import python
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

input_image = Image.open('asl_set/1/Input Images - Sign 1/Sign 1 (1).jpeg')

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
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(input_image))
    landmarks = landmarker.detect(img)

    for landmark in landmarks.hand_world_landmarks[0]:
        key_points.append([landmark.x, landmark.y, -landmark.z])

key_points = np.array(key_points)

min_z = np.min(key_points[:, 2])
max_z = np.max(key_points[:, 2])

plt.figure(figsize=(2, 2))

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
    zs = [(key_points[i, 2] - min_z) / (max_z - min_z) * 32 + 32 for i in indices]

    plt.plot(xs, ys, zorder=0, color='red', linewidth=2)
    plt.scatter(xs[1:], ys[1:], s=zs[1:], zorder=1, edgecolor='red', linewidths=2, color='lime')


plt.figimage(input_image, zorder=-1)

plt.gca().set_facecolor('black')
plt.gca().invert_yaxis()
plt.axis('off')

plt.show()
