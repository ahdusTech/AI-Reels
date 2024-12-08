import shutil
import sys
import os
import json
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
from openai import OpenAI
from pytube import YouTube
from dotenv import load_dotenv
import openai
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  #
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
openai.api_key = "openai-api-key"
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=openai.api_key)

# Parse time function (Unchanged from original)
def parse_time(time_str):
    if '.' in time_str:
        seconds_part, milliseconds_part = time_str.split('.')
        milliseconds = int(milliseconds_part)
    else:
        seconds_part, milliseconds = time_str, 0

    time_parts = list(map(int, seconds_part.split(":")))
    seconds = sum(x * 60 ** i for i, x in enumerate(reversed(time_parts)))
    return seconds + milliseconds / 1000.0

# Segment Video function (only logs segment info now)
def generate_segments(response):
    for i, segment in enumerate(response):
        print(i, segment)

        # Parse start and end times
        start_time = parse_time(segment.get("start_time", "0:00:00"))
        end_time = parse_time(segment.get("end_time", "0:00:00"))

        # Generate output file name
        output_file = f"output{str(i).zfill(3)}.mp4"

        # Construct FFmpeg command
        command = f"ffmpeg -y -i tmp/input_video.mp4 -vf scale='1920:1080' -qscale:v 3 -b:v 6000k -ss {start_time} -to {end_time} -copyts tmp/{output_file}"
        subprocess.call(command, shell=True)



# Video and Audio Processing functions (Simplified)
def load_video(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

def process_audio(input_file, output_file):
    try:
        # Extract audio from the original video
        command = f"ffmpeg -y -i tmp/{input_file} -vn -acodec copy tmp/output-audio.aac"
        subprocess.call(command, shell=True)

        # Merge audio with the processed video
        command = f"ffmpeg -y -i tmp/{output_file} -i tmp/output-audio.aac -c:v copy -c:a aac -strict experimental tmp/final-{output_file}"
        subprocess.call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing audio: {e}")

def generate_viral(transcript):
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

    prompt = f"Given the following video transcript, analyze each part for potential virality and identify 3 most viral segments from the transcript.Segment should have be one self contained reeel so 30-40 secs . The provided transcript is as follows: {transcript}. Based on your analysis, return a JSON document containing the timestamps (start and end), the description of the viral part, and its duration. The JSON document should follow this format: {json_template}. Please replace the placeholder values with the actual results from your analysis."
    system = "You are a Viral Segment Identifier, an AI system that analyzes a video's transcript and predict which segments might go viral on social media platforms. You use factors such as emotional impact, humor, unexpected content, and relevance to current trends to make your predictions. You return a structured JSON document detailing the start and end times, the description, and the duration of the potential viral segments."
    messages = [
        {"role": "system", "content" : system},
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
        model="gpt-4o-mini",  # Ensure you have access to this model in your API plan
        messages=messages,
        response_format=response_format,
        n=1,
        stop=None
    )
    # Check and adjust the response handling according to the actual method's return type
    if hasattr(response, 'choices'):
        return response.choices[0].message.content
    else:
        # This will handle cases where response structure is different or an error occurred
        raise Exception("Unexpected response structure from OpenAI API.")

def generate_subtitle(input_file, output_folder, results_folder):
    output_srt = f"{results_folder}/{output_folder}/subtitles.srt"
    if os.path.exists(output_srt):
        logging.info(f"Subtitle file {output_srt} already exists, skipping.")
        return

    command = f"auto_subtitle tmp/final-{input_file} -o {results_folder}/{output_folder} --model medium"
    logging.info(f"Running subtitle generation command: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Subtitle generation failed for {input_file}: {e}")

def generate_transcript(input_file):
    srt_file = f"tmp/{os.path.basename(input_file).split('.')[0]}.srt"

    if os.path.exists(srt_file):
        logging.info(f"SRT file {srt_file} already exists, skipping transcription generation.")
        with open(srt_file, 'r', encoding='utf-8') as file:
            return file.read()

    command = f"auto_subtitle tmp/{input_file} --srt_only True --output_srt True -o tmp/ --model medium"
    subprocess.call(command, shell=True)

    with open(srt_file, 'r', encoding='utf-8') as file:
        return file.read()

def main(input_video_path):
    if not os.path.exists(input_video_path):
        logging.error(f"File {input_video_path} does not exist")
        sys.exit(1)

    try:
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")
        os.mkdir('tmp')
    except OSError as error:
        logging.error(error)

    video_id = os.path.basename(input_video_path).split('.')[0]
    filename = 'input_video.mp4'

    command = f"cp {input_video_path} tmp/{filename}"
    subprocess.call(command, shell=True)

    output_folder = 'results'
    os.makedirs(f"{output_folder}/{video_id}", exist_ok=True)

    output_file = f"{output_folder}/{video_id}/content-bezos.txt"
    if not os.path.exists(output_file):
        transcript = generate_transcript(filename)
        viral_segments = generate_viral(transcript)
        viral_segments = json.loads(viral_segments)

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(viral_segments, file, ensure_ascii=False, indent=4)
        logging.info(f"Full transcription written to {output_file}")
    else:
        with open(output_file, 'r', encoding='utf-8') as file:
            viral_segments = json.load(file)

    generate_segments(viral_segments['segments'])

    with ThreadPoolExecutor() as executor:
        futures = []
        for i, segment in enumerate(viral_segments['segments']):
            output_file = f'output{str(i).zfill(3)}.mp4'
            futures.append(executor.submit(process_audio, filename, output_file))

        for future in as_completed(futures):
            future.result()

    for i, segment in enumerate(viral_segments['segments']):
        output_file = f'output{str(i).zfill(3)}.mp4'
        #output_file = '5dec-bexz.mp4'
        generate_subtitle(output_file, video_id, output_folder)

# Example usage
input_video_path = "/home/saad/Downloads/bezos.mp4"  # Replace with your actual video path
if __name__ == '__main__':
    main(input_video_path)
