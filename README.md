# 🎥 Video Recognition and Analysis

This project is designed to analyze video content to identify human emotions, physical activities, and generate a concise summary of the detected events.

## 🧪 Objective

The primary goal is to process a video file, perform real-time analysis on each frame to detect faces, emotions, and body poses, and then use this information to identify specific activities and anomalies. Finally, it generates a summary of all detected emotions and activities throughout the video.

## ✨ Features

- **Emotion Detection:** Utilizes `DeepFace` to identify the dominant emotion of individuals in the video (e.g., happy, sad, neutral).
- **Pose Estimation:** Employs `MediaPipe` to detect human body landmarks and analyze poses.
- **Activity Recognition:** A custom logic built on pose estimation data to identify activities such as:
    - Waving
    - Raising one or both arms
    - Touching the face
- **Anomaly Detection:** Identifies abrupt or atypical movements that deviate from normal patterns.
- **Video Output:** Generates a new video file with bounding boxes, landmarks, and labels for detected emotions, activities, and anomalies overlaid on the original footage.
- **Text Summary:** Creates a text file containing a summary of all unique emotions and activities detected, the total number of frames analyzed, and a count of detected anomalies. This summary is generated using `Hugging Face Transformers`.

## ⚙️ How it Works

1.  **Video Input:** The script takes a path to a video file as input.
2.  **Frame-by-Frame Processing:** It iterates through each frame of the video.
3.  **Analysis:** For each frame, it performs:
    - Emotion analysis using `DeepFace`.
    - Pose detection using `MediaPipe`.
    - Activity detection based on the pose landmarks.
4.  **Annotation:** The results of the analysis are drawn onto the frame.
5.  **Output Generation:** The annotated frames are compiled into a new output video, and a final summary is written to a `.txt` file.

## 🛠️ Tools and Libraries Used

- **OpenCV (`cv2`):** For video processing (reading, writing, and drawing).
- **`face_recognition`:** Used for locating faces in the video frames.
- **`DeepFace`:** For emotion analysis.
- **`MediaPipe`:** For pose estimation.
- **`Hugging Face Transformers`:** For generating the final text summary.
- **`tqdm`:** To display a progress bar during video processing.

## 🚀 How to Run

1.  **Install Dependencies:**
    Make sure you have all the required libraries installed. You can usually install them via pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Place your video:**
    Put the video you want to analyze in the `recognition-video` directory and name it `video.mp4`, or update the `input_video_path` in `main.py`.

3.  **Execute the script:**
    Run the main script from your terminal:
    ```bash
    python recognition-video/main.py
    ```

## 📤 Output

- **`output_video.mp4`:** A video file showing the original video with all the annotations.
- **`summary.txt`:** A text file containing a detailed summary including total frames analyzed, detected emotions, activities, and the number of anomalies.
