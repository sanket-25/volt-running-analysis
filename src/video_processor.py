# src/video_processor.py
import cv2
from pose_analysis import PoseAnalyzer

class VideoProcessor:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.pose_analyzer = PoseAnalyzer()

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        if cap.isOpened():
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            out = cv2.VideoWriter(self.output_path, fourcc, 30, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Pose detection and annotation
            landmarks = self.pose_analyzer.detect_pose(frame)
            frame = self.pose_analyzer.draw_landmarks(frame, landmarks)

            # Write the frame to output
            out.write(frame)
            cv2.imshow('Running Analysis', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

