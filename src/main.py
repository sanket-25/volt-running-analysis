import os
import cv2
import tensorflow as tf
from video_processor import VideoProcessor
from pose_analysis import PoseAnalyzer

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings

if __name__ == "__main__":
    input_video = 'data/video.mp4'
    output_video = 'results/output_video.mp4'

    processor = VideoProcessor(input_video, output_video)

    # Analyze video and collect pose data
    processor.process_video()

    # Example to extract and analyze one frame (for demo purposes)
    pose_analyzer = PoseAnalyzer()

    # Process a single frame for inference (for demo)
    cap = cv2.VideoCapture(input_video)
    ret, frame = cap.read()
    if ret:
        landmarks = pose_analyzer.detect_pose(frame)
        stride_length = pose_analyzer.calculate_stride_length(landmarks)
        knee_drive = pose_analyzer.calculate_knee_drive(landmarks)
        torso_angle = pose_analyzer.calculate_torso_angle(landmarks)

        # Get feedback on running technique
        feedback = pose_analyzer.give_feedback(stride_length, knee_drive, torso_angle)

        # Output feedback
        if feedback:
            print("Feedback on running technique:")
            for suggestion in feedback:
                print(f"- {suggestion}")
        else:
            print("Your running technique looks good!")
