import random
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import json
import subprocess
import logging
from pathlib import Path
import openai
from dotenv import load_dotenv
import cv2
import numpy as np

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
openai.api_key = "projects/557964912809/secrets/ai-reels-secret"
if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please check your environment variables.")
else:
    logging.info(f"OpenAI API key loaded from environment variable")
MODEL = "gpt-4o-mini"
client = openai.OpenAI(api_key=openai.api_key)

# Directory for storing uploaded and processed files
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    logging.info(f"Received upload request for file: {file.filename}")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"File {file.filename} saved to {file_path}")

        # Process the video
        process_video(file_path)
        logging.info(f"Video {file.filename} processed successfully")
    except Exception as e:
        logging.error(f"Error processing video {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"filename": file.filename, "message": "Video processed successfully."}


@app.get("/download/{video_id}")
async def download_video(video_id: str):
    result_path = os.path.join(RESULTS_DIR, video_id)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Video not found")

    result_file = next(Path(result_path).glob("*.mp4"), None)
    if result_file:
        return FileResponse(result_file)
    else:
        raise HTTPException(status_code=404, detail="Processed video not found")


@app.get("/videos")
async def list_videos():
    return [f.name for f in Path(RESULTS_DIR).iterdir() if f.is_dir()]


import os
import json
import subprocess
import logging

RESULTS_DIR = "./results"  # Change this to your desired results directory


def process_video(input_video_path):
    logging.info(f"Starting video processing for {input_video_path}")
    video_id = os.path.basename(input_video_path).split('.')[0]
    output_folder = os.path.join(RESULTS_DIR, video_id)
    os.makedirs(output_folder, exist_ok=True)
    logging.debug(f"Output folder created at {output_folder}")

    try:
        # Example processing logic
        transcript = generate_transcript(input_video_path)
        logging.debug(f"Transcript generated for {input_video_path}")

        viral_segments = generate_viral(transcript, video_id)
        logging.debug(f"Viral segments generated for {video_id}")

        viral_segments = json.loads(viral_segments)

        for i, segment in enumerate(viral_segments['segments']):
            logging.info(f"Processing segment {i} for video {video_id}")
            start_time = parse_time(segment.get("start_time", "0:00:00"))
            end_time = parse_time(segment.get("end_time", "0:00:00"))

            # Output file for the current segment
            output_file = os.path.join(output_folder, f"output{str(i).zfill(3)}.mp4")
            logging.debug(f"Output file path: {output_file}")

            # Extract video segment
            command = f"ffmpeg -y -i {input_video_path} -vf scale='1920:1080' -qscale:v 3 -b:v 6000k -ss {start_time} -to {end_time} -copyts {output_file}"
            subprocess.call(command, shell=True)
            logging.info(f"Video segment {i} extracted to {output_file}")

            # Prepare paths for further processing
            temp_video_path = os.path.join(output_folder, f"temp_output_video_{str(i).zfill(3)}.mp4")
            audio_path = os.path.join(output_folder, f"extracted_audio_{str(i).zfill(3)}.aac")
            final_output_video_path = os.path.join(output_folder, f"final_video_{str(i).zfill(3)}.mp4")
            final_with_subtitles_path = os.path.join(output_folder, f"final_video_with_subtitles_{str(i).zfill(3)}.mp4")

            # Video processing
            video_file_specs = preprocess_video(output_file)
            resolution_needs = resolution_specs("reels")
            process_video_with_dynamic_padding(output_file, resolution_needs, temp_video_path, video_file_specs)

            # Extract audio
            extract_audio(output_file, audio_path)

            # Combine video and audio
            combine_video_audio(temp_video_path, audio_path, final_output_video_path)
            #add_subtitles(segment, transcript, final_output_video_path, final_with_subtitles_path)
            logging.info(f"Video segment {i} processing complete with subtitles: {final_with_subtitles_path}")

    except Exception as e:
        logging.error(f"Error during video processing: {e}")
        raise


def add_subtitles(segment, transcript, final_output_video_path, final_with_subtitles_path):
    generate_transcript(final_output_video_path)
    # Create a temporary subtitle file for the segment
    segment_srt_path = f"{UPLOAD_DIR}/{os.path.basename(final_output_video_path).split('.')[0]}.srt"

    with open(segment_srt_path, 'w', encoding='utf-8') as srt_file:
        srt_file.write(transcript)

    # Use ffmpeg to burn subtitles into the video
    command = (
        f"ffmpeg -y -i {final_output_video_path} -vf subtitles={segment_srt_path} "
        f"-c:v libx264 -c:a copy {final_with_subtitles_path}"
    )
    subprocess.call(command, shell=True)

    # Log completion
    logging.info(f"Added subtitles to video segment, saved to {final_with_subtitles_path}")

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


def resolution_specs(format_type):
    logging.info(f"Getting resolution specs for format type: {format_type}")
    if format_type == "reels":
        return {'height': 1920, 'width': 1080}
    else:
        return {'height': 1080, 'width': 1920}

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


def resize_with_dynamic_padding(frame, resolution_needs, background, use_gaussian_blur=True):
    original_height, original_width = frame.shape[:2]
    aspect_ratio = original_width / original_height
    new_width = resolution_needs['width']
    new_height = int(new_width / aspect_ratio)

    if new_height > resolution_needs['height']:
        new_height = resolution_needs['height']
        new_width = int(new_height * aspect_ratio)

    # Resize the frame to fit within the required resolution
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Choose background based on the flag
    if use_gaussian_blur:
        static_background = background.copy()  # Gaussian blurred frame
    else:
        static_background = background  # Random color background

    # Calculate the padding to center the resized frame
    top_padding = (resolution_needs['height'] - new_height) // 2
    bottom_padding = resolution_needs['height'] - new_height - top_padding
    left_padding = (resolution_needs['width'] - new_width) // 2
    right_padding = resolution_needs['width'] - new_width - left_padding

    # Overlay the resized frame onto the background
    static_background[top_padding:top_padding + new_height,
                      left_padding:left_padding + new_width] = resized_frame

    return static_background


def process_video_with_dynamic_padding(video_file_path, resolution_needs, temp_video_path, video_file_specs, use_gaussian_blur=True):
    logging.info("Processing video segments")

    # Define video writer with required resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, video_file_specs['fps'],
                          (resolution_needs['width'], resolution_needs['height']))

    # Open the input video file
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        logging.error("Video cannot be opened")
        raise ValueError("Video cannot be opened")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare background
    if use_gaussian_blur:
        # Select a random frame for Gaussian blur
        random_frame_index = random.randint(0, total_frames - 1)
        logging.info(f"Selected frame {random_frame_index} for Gaussian blur background")

        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)
        ret, random_frame = cap.read()
        if not ret:
            logging.error("Failed to read the selected random frame for Gaussian blur background")
            raise ValueError("Failed to read the selected random frame")

        # Create a Gaussian blurred background from the selected frame
        background = cv2.GaussianBlur(random_frame, (51, 51), 0)
        background = cv2.resize(background,
                                (resolution_needs['width'], resolution_needs['height']))
    else:
        # Generate a random color background
        random_color = [random.randint(0, 255) for _ in range(3)]
        background = np.full((resolution_needs['height'], resolution_needs['width'], 3), random_color, dtype=np.uint8)
        logging.info(f"Selected random color background: {random_color}")

    # Reset video capture to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Frame read failed at frame {frame_idx}")
            break

        # Resize the frame with the chosen background
        padded_frame = resize_with_dynamic_padding(frame, resolution_needs, background, use_gaussian_blur)

        # Write the processed frame to the output video
        out.write(padded_frame)

    # Release resources
    cap.release()
    out.release()
    logging.info(f"Temporary processed video saved to {temp_video_path}")



def parse_time(time_str):
    if '.' in time_str:
        seconds_part, milliseconds_part = time_str.split('.')
        milliseconds = int(milliseconds_part)
    else:
        seconds_part, milliseconds = time_str, 0

    time_parts = list(map(int, seconds_part.split(":")))
    seconds = sum(x * 60 ** i for i, x in enumerate(reversed(time_parts)))
    return seconds + milliseconds / 1000.0


def generate_transcript(input_file):
    srt_file = f"{UPLOAD_DIR}/{os.path.basename(input_file).split('.')[0]}.srt"

    # Check if the subtitle file already exists
    if os.path.exists(srt_file):
        logging.info(f"Subtitle file {srt_file} already exists. Using existing file.")
    else:
        logging.info(f"Generating subtitle file for {input_file}.")
        command = f"auto_subtitle {input_file} --srt_only True --output_srt True -o {UPLOAD_DIR}/ --model medium"
        subprocess.call(command, shell=True)
        logging.info(f"Subtitle file {srt_file} generated.")
    # Read and return the subtitle content
    try:
        with open(srt_file, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"Subtitle file {srt_file} not found.")
        raise HTTPException(status_code=404, detail="Subtitle file not found")



def generate_viral(transcript, video_id):
    metadata_file = f"{RESULTS_DIR}/{video_id}_metadata.json"

    # Check if the metadata file already exists
    if os.path.exists(metadata_file):
        logging.info(f"Metadata file {metadata_file} already exists. Using existing file.")
        with open(metadata_file, 'r', encoding='utf-8') as file:
            return file.read()

    if not client:
        raise ValueError("OpenAI API key is not set. Please check your environment variables.")

    json_template = '''
        { "segments" :
            [
                {
                    "start_time": 00.00, 
                    "end_time": 00.00,
                    "description": "Description of the text",
                    "duration":00,
                },    
            ]
        }
    '''

    prompt = f"Given the following video transcript, analyze each part for potential virality and identify 3 most viral segments from the transcript. Segment should be one self-contained reel so 20-40 secs maximum limit. The provided transcript is as follows: {transcript}. Based on your analysis, return a JSON document containing the timestamps (start and end), the description of the viral part, and its duration. The JSON document should follow this format: {json_template}. Please replace the placeholder values with the actual results from your analysis."
    system = "You are a Viral Segment Identifier, an AI system that analyzes a video's transcript and predicts which segments might go viral on social media platforms. You use factors such as emotional impact, humor, unexpected content, and relevance to current trends to make your predictions. You return a structured JSON document detailing the start and end times, the description, and the duration of the potential viral segments."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "viral_segments_identification",
            "schema": {
                "type": "object",
                "properties": {
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_time": {"type": "string"},
                                "end_time": {"type": "string"},
                                "description": {"type": "string"},
                                "duration": {"type": "number"}
                            },
                            "required": ["start_time", "end_time", "description", "duration"],
                            "additionalProperties": False
                        }
                    },
                    "note": {"type": "string"}
                },
                "required": ["segments", "note"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format=response_format,
        n=1,
        stop=None
    )
    if hasattr(response, 'choices'):
        metadata_content = response.choices[0].message.content
        # Save the metadata to a file
        with open(metadata_file, 'w', encoding='utf-8') as file:
            file.write(metadata_content)
        return metadata_content
    else:
        raise Exception("Unexpected response structure from OpenAI API.")


