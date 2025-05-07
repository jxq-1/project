import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_video(video_path, segment_duration):
    try:
        video = VideoFileClip(video_path)
        total_duration = video.duration
        segment_count = int(total_duration // segment_duration)

        segments = []
        for i in range(segment_count):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            segment = video.subclip(start_time, end_time)
            segments.append(segment)

        return segments
    except Exception as e:
        logging.error(f"Error splitting video: {e}")
        return []

def extract_frames(segments, output_folder):
    try:
        frames_folder = os.path.join(output_folder, "frames")
        os.makedirs(frames_folder, exist_ok=True)

        frame_files = []
        for i, segment in enumerate(segments):
            for t in range(int(segment.duration)):
                global_frame_index = i * segment.duration + t
                group_number = global_frame_index // 10
                frame_path = f"{frames_folder}/group_{group_number}_frame_{global_frame_index:04d}.png"
                segment.save_frame(frame_path, t)
                frame_files.append(frame_path)
        
        return frames_folder
    except Exception as e:
        logging.error(f"Error extracting frames: {e}")
        return None

def extract_audio(segments, output_folder):
    try:
        audio_folder = os.path.join(output_folder, "audio")
        os.makedirs(audio_folder, exist_ok=True)

        audio_paths = []
        for i, segment in enumerate(segments):
            audio_path = f"{audio_folder}/audio_{i}.wav"
            segment.audio.write_audiofile(audio_path)
            audio_paths.append(audio_path)
        
        return audio_folder
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        return None

def save_videos(segments, output_folder):
    try:
        video_folder = os.path.join(output_folder, "videos")
        os.makedirs(video_folder, exist_ok=True)

        video_paths = []
        for i, segment in enumerate(segments):
            video_path = f"{video_folder}/video_{i}.mp4"
            segment.write_videofile(video_path, codec='libx264')
            video_paths.append(video_path)
        
        return video_folder
    except Exception as e:
        logging.error(f"Error saving videos: {e}")
        return None

def main():
    video_path = "math_class.mp4"  # 输入视频文件的路径
    segment_duration = 10  # 每个子视频的持续时间（秒）
    output_folder = "math_class"  # 存放子视频、帧和音频的文件夹路径

    if not os.path.exists(video_path):
        logging.error("Video file does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    segments = split_video(video_path, segment_duration)

    frames_folder = extract_frames(segments, output_folder)
    audio_folder = extract_audio(segments, output_folder)
    video_folder = save_videos(segments, output_folder)

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()