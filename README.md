# ASLDigitClassification

Used the MediaPipe framework to note key points in the hands gestures and then develop a model that classifies those hand gestures as the digits 0-9.

## Dataset

[MediaPipe_Processed_ASL_Dataset](https://www.kaggle.com/datasets/vignonantoine/mediapipe-processed-asl-dataset)

## Model

1. Load image input tensor
2. Create output tensor
3. Split train and test
4. Train on CNN
5. Run on test data
