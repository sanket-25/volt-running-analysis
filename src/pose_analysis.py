# src/pose_analysis.py (extended with feedback)
import cv2
import mediapipe as mp
import numpy as np

class PoseAnalyzer:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()

    def detect_pose(self, frame):
        """Detects pose landmarks in a video frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results.pose_landmarks

    def draw_landmarks(self, frame, landmarks):
        """Draw landmarks on the video frame."""
        mp_drawing = mp.solutions.drawing_utils
        if landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return frame

    def calculate_stride_length(self, landmarks):
        """Calculate stride length based on the distance between left and right ankles."""
        if landmarks:
            left_ankle = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
            # Euclidean distance between left and right ankle
            stride_length = np.sqrt((left_ankle.x - right_ankle.x) ** 2 + (left_ankle.y - right_ankle.y) ** 2)
            return stride_length
        return 0

    def calculate_knee_drive(self, landmarks):
        """Calculate the height of the knee relative to the hip."""
        if landmarks:
            left_knee = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            left_hip = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            knee_drive = left_knee.y - left_hip.y
            return knee_drive
        return 0

    def calculate_torso_angle(self, landmarks):
        """Calculate torso lean angle using shoulders and hips."""
        if landmarks:
            left_shoulder = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

            # Average shoulder and hip position
            shoulder_mid = (left_shoulder.y + right_shoulder.y) / 2
            hip_mid = (left_hip.y + right_hip.y) / 2

            # Calculate the angle
            torso_angle = np.degrees(np.arctan2(shoulder_mid - hip_mid, 1))  # Simplified to 1 unit for x-axis distance
            return torso_angle
        return 0

    def give_feedback(self, stride_length, knee_drive, torso_angle):
        """Provide basic feedback based on metrics."""
        feedback = []

        # Stride Length Feedback
        if stride_length < 0.2:  # Threshold for under-striding
            feedback.append("Increase your stride length for better efficiency.")

        # Knee Drive Feedback
        if knee_drive < 0.1:  # Threshold for low knee drive
            feedback.append("Increase your knee lift for more power.")

        # Torso Angle Feedback
        if torso_angle < -15:  # Too much forward lean
            feedback.append("You're leaning too far forward. Straighten your torso.")
        elif torso_angle > 15:  # Leaning backward
            feedback.append("You're leaning too far backward. Lean slightly forward.")

        return feedback
