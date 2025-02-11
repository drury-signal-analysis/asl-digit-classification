import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import numpy as np

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

for digit in range(10):
    for i in range(500):
        print(f'{digit}, {i + 1}')
        image = cv.imread(f'asl_set/{digit}/Input Images - Sign {digit}/Sign {digit} ({i + 1}).jpeg')

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            print('No landmarks')
            continue

        output_image = np.zeros(image.shape)

        for hand_landmarks in results.multi_hand_landmarks:
            drawing.draw_landmarks(
                output_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

        cv.imwrite(f'images/{digit}/{i}.png', output_image)
