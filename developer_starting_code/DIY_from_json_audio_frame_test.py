import cv2
import json
import numpy as np
import logging
import sys
import gc
import subprocess
"""This code converts the video into narrator and non narrator"""
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("video_processing.log")
    ]
)

def resize_with_padding(frame, resolution_needs):
    original_height, original_width = frame.shape[:2]
    aspect_ratio = original_width / original_height
    new_width = resolution_needs['width']
    new_height = int(new_width / aspect_ratio)

    if new_height > resolution_needs['height']:
        new_height = resolution_needs['height']
        new_width = int(new_height * aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))

    top_padding = (resolution_needs['height'] - new_height) // 2
    bottom_padding = resolution_needs['height'] - new_height - top_padding
    left_padding = (resolution_needs['width'] - new_width) // 2
    right_padding = resolution_needs['width'] - new_width - left_padding

    padded_frame = cv2.copyMakeBorder(
        resized_frame,
        top_padding, bottom_padding,
        left_padding, right_padding,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return padded_frame

def read_json(file_path):
    logging.info(f"Reading JSON file from {file_path}")
    with open(file_path, 'r') as file:
        return json.load(file)

def preprocess_video(video_path):
    logging.info(f"Preprocessing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Video cannot be opened")
        raise ValueError("Video cannot be opened")

    # Extract video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    logging.info(f"Video properties: Width={width}, Height={height}, FPS={fps}, Codec={codec}")

    cap.release()

    return {'width': width, 'height': height, 'fps': fps, 'codec': codec}

def resolution_specs(format_type):
    logging.info(f"Getting resolution specs for format type: {format_type}")
    if format_type == "reels":
        return {'height': 1920, 'width': 1080}
    else:
        return {'height': 1080, 'width': 1920}


def process_narrator_segment(cap, start_frame, end_frame, out, video_file_specs, resolution_needs):
    # Calculate scaling factor for narrator segments
    scale = resolution_needs['height'] / video_file_specs['height']
    new_width = int(video_file_specs['width'] * scale)
    cropped_width = resolution_needs['width']

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Frame read failed at narrator segment")
            break

        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, resolution_needs['height']))

        # Center crop
        start_x = (new_width - cropped_width) // 2
        cropped_frame = resized_frame[:, start_x:start_x + cropped_width]

        out.write(cropped_frame)
def process_video_segments(video_file_specs, json_file, video_path, resolution_needs, temp_video_path):
    logging.info("Processing video segments")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, video_file_specs['fps'], (resolution_needs['width'], resolution_needs['height']))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Video cannot be opened")
        raise ValueError("Video cannot be opened")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_segments = []

    # Combine narrator and non-narrator segments with type tags
    for segment in json_file.get('narrator_segments', []):
        frame_segments.append((segment['start_time'], segment['end_time'], 'narrator'))

    for segment in json_file.get('non_narrator_segments', []):
        frame_segments.append((segment['start_time'], segment['end_time'], 'non_narrator'))

    # Sort segments by start time
    frame_segments.sort(key=lambda x: x[0])

    last_end_frame = 0

    for start_time, end_time, segment_type in frame_segments:
        start_frame = int(start_time * video_file_specs['fps'])
        end_frame = int(end_time * video_file_specs['fps'])

        # Process uncategorized frames as non-narrator
        if start_frame > last_end_frame:
            for frame_idx in range(last_end_frame, start_frame):
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Frame read failed at frame {frame_idx}")
                    break
                padded_frame = resize_with_padding(frame, resolution_needs)
                out.write(padded_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        if segment_type == 'narrator':
            process_narrator_segment(cap, start_frame, end_frame, out, video_file_specs, resolution_needs)
        else:
            while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Frame read failed at segment {segment_type}")
                    break
                padded_frame = resize_with_padding(frame, resolution_needs)
                out.write(padded_frame)

        last_end_frame = end_frame

    # Process any remaining uncategorized frames at the end
    for frame_idx in range(last_end_frame, total_frames):
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Frame read failed at frame {frame_idx}")
            break
        padded_frame = resize_with_padding(frame, resolution_needs)
        out.write(padded_frame)

    cap.release()
    out.release()
    logging.info(f"Temporary processed video saved to {temp_video_path}")




def extract_audio(video_path, audio_path):
    logging.info(f"Extracting audio from {video_path}")
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y"
    subprocess.run(command, shell=True, check=True)
    logging.info(f"Audio extracted to {audio_path}")

def combine_video_audio(temp_video_path, audio_path, output_video_path):
    logging.info(f"Combining video {temp_video_path} with audio {audio_path}")
    command = f"ffmpeg -i {temp_video_path} -i {audio_path} -c:v copy -c:a aac {output_video_path} -y"
    subprocess.run(command, shell=True, check=True)
    logging.info(f"Final video saved to {output_video_path}")

def main():
    logging.info("Starting video processing")
    json_file_path = "/home/saad/Documents/saad_personal/one_stream_reels/pythonProject2/gitcode_opy/narrator_segments_01_31.json"
    video_path = "/home/saad/Documents/saad_personal/one_stream_reels/pythonProject2/segment_recognition/onestrea_vide.mp4"
    temp_video_path = "temp_output_video.mp4"
    audio_path = "extracted_audio.aac"
    output_video_path = "full_video_narrartor_nonnarrator_with_aspect.mp4"

    try:
        json_data = read_json(json_file_path)
        video_file_specs = preprocess_video(video_path)
        resolution_needs = resolution_specs("reels")

        process_video_segments(video_file_specs, json_data, video_path, resolution_needs, temp_video_path)

        extract_audio(video_path, audio_path)
        combine_video_audio(temp_video_path, audio_path, output_video_path)

        logging.info("Video processing complete.")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
