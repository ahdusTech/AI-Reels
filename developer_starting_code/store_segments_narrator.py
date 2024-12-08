import cv2
import face_recognition
import logging
import json
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
"""narrator and non narrator .json split has been acheived!"""
def detect_faces_face_recognition(video_path, interval=0.1):
    logging.info(f"Detecting faces in video: {video_path}")

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps

    face_timestamps = []
    main_narrator_encoding = None

    logging.info(f"Video duration: {duration} seconds, FPS: {fps}")

    for i in range(0, int(duration / interval)):
        current_time = i * interval  # Time in seconds
        video.set(cv2.CAP_PROP_POS_FRAMES, current_time * fps)
        ret, frame = video.read()
        if not ret:
            break

        logging.info(f"Processing frame at {current_time:.1f} seconds.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            for face_encoding in face_encodings:
                if main_narrator_encoding is None:
                    main_narrator_encoding = face_encoding
                    face_timestamps.append((current_time, "narrator"))
                else:
                    match = face_recognition.compare_faces([main_narrator_encoding], face_encoding, tolerance=0.6)
                    if match[0]:
                        face_timestamps.append((current_time, "narrator"))
                    else:
                        face_timestamps.append((current_time, "non-narrator"))
        else:
            face_timestamps.append((current_time, "non-narrator"))

    video.release()
    return face_timestamps

def process_timestamps(face_timestamps, threshold=0.4):
    logging.info("Processing timestamps to create segments.")
    narrator_segments = []
    non_narrator_segments = []

    if not face_timestamps:
        logging.warning("No face timestamps found. Returning empty segments.")
        return [], []

    current_segment = {"type": face_timestamps[0][1], "start_time": face_timestamps[0][0]}

    for i in range(1, len(face_timestamps)):
        if face_timestamps[i][1] != current_segment["type"]:
            end_time = face_timestamps[i][0]
            if current_segment["type"] == "non-narrator" and (end_time - current_segment["start_time"]) < threshold:
                # If non-narrator segment is less than the threshold, it's ignored
                current_segment["type"] = "narrator"
            else:
                current_segment["end_time"] = end_time
                if current_segment["type"] == "narrator":
                    narrator_segments.append(current_segment)
                else:
                    non_narrator_segments.append(current_segment)
                current_segment = {"type": face_timestamps[i][1], "start_time": face_timestamps[i][0]}

    # Final segment
    if "end_time" not in current_segment:
        current_segment["end_time"] = face_timestamps[-1][0]
        if current_segment["type"] == "narrator":
            narrator_segments.append(current_segment)
        else:
            non_narrator_segments.append(current_segment)

    return narrator_segments, non_narrator_segments

def save_segments_to_json(narrator_segments, non_narrator_segments, output_json):
    logging.info(f"Saving segments to {output_json}.")
    segments = {
        "narrator_segments": narrator_segments,
        "non_narrator_segments": non_narrator_segments
    }

    with open(output_json, 'w') as json_file:
        json.dump(segments, json_file, indent=4)

    logging.info("Segments saved to JSON.")

def main(video_path, output_json):
    face_timestamps = detect_faces_face_recognition(video_path)

    if not face_timestamps:
        logging.error("No faces detected in the video. Exiting.")
        return

    narrator_segments, non_narrator_segments = process_timestamps(face_timestamps)
    save_segments_to_json(narrator_segments, non_narrator_segments, output_json)

if __name__ == '__main__':
    video_path = "/home/saad/Documents/saad_personal/one_stream_reels/pythonProject2/segment_recognition/onestrea_vide.mp4"
    output_json = "narrator_segments_01_31.json"

    main(video_path, output_json)
