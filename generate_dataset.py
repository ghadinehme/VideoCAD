import cv2
import numpy as np
import os
from utils import *
from tqdm import tqdm
import pickle as pkl
from transform_dataset import convert_logs_to_vectors, process_logs
import shutil
import lmdb
import io
from PIL import Image


def extract_frame_at_timestamp(cap, timestamp):
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Compute the frame index closest to the given timestamp
    frame_idx = int(timestamp) #int(round(timestamp * fps))
    
    # Set the video position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # Read the frame at that position
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame at timestamp {timestamp} sec (frame {frame_idx}).")
        return False
    
    return frame


def extract_frames_from_actions(video_path, timestamps, resize=None):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print("Error opening video file.")
        return False
    # Retrieve the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("FPS value is 0. Check the video file.")
        return False
    
    for timestep in tqdm(timestamps):
        frame = extract_frame_at_timestamp(cap, timestep)
        if frame is False:
            print(f"Could not extract frame at timestamp {timestep} sec")
            return False
        if resize is not None:
            frame = Image.fromarray(frame)
            frame = frame.resize(resize, Image.Resampling.BILINEAR)
            frame = np.array(frame)

        frames.append(frame)

    cap.release()
    assert len(frames) == len(timestamps), "Number of frames and actions must be the same"
    return np.array(frames)

def generate_save_path(save_path, id, ext, file_type="frames"):
    tmp = id[:4]
    save_dir = os.path.join(save_path, tmp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file_type:
        save_path = os.path.join(save_dir, f"{id}_{file_type}.{ext}")
    return save_path

def save_arr_to_pkl(frames, save_path, id, file_type="frames"):
    path = generate_save_path(
        save_path, id, "pkl", file_type)
    with open(path, "wb") as f:
        pkl.dump(frames, f)

def convert_vid_to_frames(root_dir, save_dir : str):
    """
    Convert videos to frames and save them in a pickle file.
    """
    vid_dir = f"{root_dir}/videos"
    video_files = os.listdir(vid_dir)
    js_dir = f"{root_dir}/mouse_json"
    json_files = os.listdir(js_dir)
    video_files.sort()
    json_files.sort()
    vids = set([os.path.splitext(vid_file)[0] for vid_file in video_files])
    invalid_videos = []
    errored_out_videos = []

    for vid_file, json_file in zip(video_files, json_files):
        vid_split = os.path.splitext(vid_file)[0]
        js_split = os.path.splitext(json_file)[0]
        if js_split not in vids:
            continue

        vid_full = os.path.join(vid_dir, vid_file)
        js_full = os.path.join(js_dir, json_file)
        print(js_full)
        js = load_json(js_full)
        try:
            res = extract_frames_from_actions(vid_full, js)
            if res is False:
                invalid_videos.append(res)
                break
            save_arr_to_pkl(res, save_dir, js_split)
        except Exception as e:
            print(f"Error: video {vid_split}",e)
            errored_out_videos.append(vid_split)

    print(f"Invalid videos: {invalid_videos}")
    print(f"Errored out videos: {errored_out_videos}")


def generate_action_vectors_and_video_pairs(root_dir, save_dir, resize=(224, 224)):
    """
    Generate sine-transformed action vectors and their corresponding video frame sequences.
    
    For each matching pair of video and JSON files in the root_dir (expected under
    "videos" and "mouse_json" subdirectories respectively), this function will:
      1. Load the JSON action data using load_json().
      2. Extract video frames using extract_frames_from_actions().
      3. Generate two action vectors (e.g., one for absolute and one for relative coordinates)
         by calling sin_transform_dataset() on the JSON data.
    
    Note:
      - The function sin_transform_dataset(actions_data) is assumed to exist in transform_dataset.py.
        It should accept the loaded JSON action data and return a tuple (abs_vector, rel_vector),
        where abs_vector and rel_vector represent sine-transformed versions of the action data.
      - Similarly, load_json() is assumed to be defined in the scope.
    
    Args:
        root_dir (str): The root directory containing:
                         - a "videos" folder with video files, and
                         - a "mouse_json" folder with corresponding JSON files.
    
    Returns:
        tuple: (action_vectors, video_pairs)
               where action_vectors is a list of tuples, each tuple being (abs_vector, rel_vector)
               for one video, and video_pairs is a list of numpy arrays containing extracted frames.
    """

    video_dir = os.path.join(root_dir, "videos")
    log_dir = os.path.join(root_dir, "mouse")
    image_dir = os.path.join(root_dir, "images")

    video_files = sorted(os.listdir(video_dir))

    for video_file in video_files:
        video_base = os.path.splitext(video_file)[0]
        log_file = f"{video_base}.log"
        log_path = os.path.join(log_dir, log_file)
        image_file = f"{video_base}_0.png"
        image_path = os.path.join(image_dir,video_base[:4], image_file)

        # Check if the corresponding JSON file exists
        if not os.path.exists(log_path):
            print(f"Warning: No matching JSON file for {video_file}")
            continue
        if not os.path.exists(image_path):
            print(f"Warning: No matching image file for {image_path}")
            continue

        target_path = generate_save_path(save_dir, video_base, "pkl", "data")
        save_path = generate_save_path(save_dir, video_base, "png")
        vid_path = os.path.join(video_dir, video_file)
        print(f"Processing {vid_path} and {log_path}")
        if os.path.exists(target_path):
            print(f"Skipping {video_file} because it already exists")
            continue
        
        

        # Load the JSON actions data (assumed to be a list of dictionaries)
        actions_data = process_logs(open_file(log_path), False)
        # Generate the sine-transformed action vectors using the function from transform_dataset.py
        actions, timesteps = convert_logs_to_vectors(actions_data)
        # Extract video frames based on the action list
        frames = extract_frames_from_actions(vid_path, timesteps, resize)
        if frames is False:
            continue
        zeros = np.zeros((1, 7))
        frames = np.vstack([frames[:1], frames])
        actions = np.vstack([zeros, actions])
        timesteps = np.array(timesteps[:1]+timesteps)

        end_action = 950
        end_idx = np.where(actions[:, 3] == end_action)[0]
        if len(end_idx) > 0:
            print(f"End idx: {end_idx}")
            actions = actions[:end_idx[0]+1]
            frames = frames[:end_idx[0]+1]
            timesteps = timesteps[:end_idx[0]+1]

        assert len(frames) == len(actions), "Number of frames and actions must be the same"
        data = {"frames": frames, 
                "actions": actions, 
                "timesteps": timesteps}
        
        
        save_to_pkl(data, target_path)

        
        shutil.copy(image_path, save_path)



if __name__ == "__main__":
    generate_action_vectors_and_video_pairs("data/data_raw", 
                                            "data/data_resized",
                                            resize=(224, 224))
