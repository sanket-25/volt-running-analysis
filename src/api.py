import os
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pose_analysis import PoseAnalyzer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def hello_world():
    return jsonify({"message": "Hello, World!"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)

    # Save the uploaded file
    file.save(filepath)

    # Log the file paths
    print(f"Input video saved at: {os.path.abspath(filepath)}")
    print(f"Processed video will be saved at: {os.path.abspath(processed_filepath)}")

    # Process the video using the PoseAnalyzer
    pose_analyzer = PoseAnalyzer()
    cap = cv2.VideoCapture(filepath)

    # Open the input video and create a VideoWriter to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(processed_filepath, fourcc, fps, (width, height))

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect pose in each frame
        landmarks = pose_analyzer.detect_pose(frame)
        
        # Optional: draw pose landmarks on the frame (for visualization purposes)
        frame_with_landmarks = pose_analyzer.draw_landmarks(frame, landmarks)
        
        # Write the processed frame to the output video
        out.write(frame_with_landmarks)

    cap.release()
    out.release()

    # Check if the processed file has been created
    if not os.path.exists(processed_filepath):
        print(f"Error: Processed video not found at {processed_filepath}")
        return jsonify({"error": "Processed video could not be created."}), 500

    # Log the final processed file path
    print(f"Processed video saved at: {os.path.abspath(processed_filepath)}")

    # Return the processed video as a downloadable file along with feedback
    return send_file(os.path.abspath(processed_filepath), as_attachment=True, mimetype='video/mp4', download_name=processed_filename), 200

if __name__ == '__main__':
    app.run(debug=True)
