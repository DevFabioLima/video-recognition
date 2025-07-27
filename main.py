import cv2
import face_recognition
from tqdm import tqdm
from deepface import DeepFace
import mediapipe as mp
from transformers import pipeline

def analyze_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    detected_emotions = []
    detected_activities = []

    prev_left_hand_x = 0
    prev_right_hand_x = 0
    wave_frames = 0

    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face recognition and emotion analysis
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            for face in result:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                dominant_emotion = face['dominant_emotion']
                detected_emotions.append(dominant_emotion)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        except Exception as e:
            print(f"Could not analyze frame for emotion: {e}")

        # Pose detection
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Activity detection logic
            landmarks = pose_results.pose_landmarks.landmark
            
            # Left arm up
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_arm_up = left_wrist.y < left_elbow.y < left_shoulder.y

            # Right arm up
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_arm_up = right_wrist.y < right_elbow.y < right_shoulder.y

            if left_arm_up and right_arm_up:
                activity = "Ambos os bracos levantados"
                detected_activities.append(activity)
                cv2.putText(frame, activity, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif left_arm_up:
                activity = "Braco esquerdo levantado"
                detected_activities.append(activity)
                cv2.putText(frame, activity, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif right_arm_up:
                activity = "Braco direito levantado"
                detected_activities.append(activity)
                cv2.putText(frame, activity, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Hand on face
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_wrist_on_face = abs(left_wrist.x - nose.x) < 0.1 and abs(left_wrist.y - nose.y) < 0.1
            right_wrist_on_face = abs(right_wrist.x - nose.x) < 0.1 and abs(right_wrist.y - nose.y) < 0.1

            if left_wrist_on_face or right_wrist_on_face:
                activity = "Mao no rosto"
                detected_activities.append(activity)
                cv2.putText(frame, activity, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Waving (bye)
            if left_arm_up or right_arm_up:
                if prev_left_hand_x == 0:
                    prev_left_hand_x = left_wrist.x
                if prev_right_hand_x == 0:
                    prev_right_hand_x = right_wrist.x

                if (abs(left_wrist.x - prev_left_hand_x) > 0.05) or \
                   (abs(right_wrist.x - prev_right_hand_x) > 0.05):
                    wave_frames += 1
                else:
                    wave_frames = 0

                if wave_frames > 5: # 5 consecutive frames of waving
                    activity = "Tchau"
                    detected_activities.append(activity)
                    cv2.putText(frame, activity, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            prev_left_hand_x = left_wrist.x
            prev_right_hand_x = right_wrist.x


        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return detected_emotions, detected_activities

def summarize_text(text, max_length=730, min_length=30, do_sample=False):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
    return summary[0]['summary_text']

if __name__ == "__main__":
    input_video_path = '/Users/fabio.lima/Workspace/AI/recognition-video/video.mp4'
    output_video_path = '/Users/fabio.lima/Workspace/AI/recognition-video/output_video.mp4'
    
    emotions, activities = analyze_video(input_video_path, output_video_path)
    
    summary_text = f"Detected emotions: {', '.join(list(set(emotions)))}. Detected activities: {', '.join(list(set(activities)))}."
    
    final_summary = summarize_text(summary_text)
    
    print("Resumo gerado:")
    print(final_summary)

    with open('/Users/fabio.lima/Workspace/AI/recognition-video/summary.txt', 'w') as f:
        f.write(final_summary)
