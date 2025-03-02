import os

dataset_dir = "I:/wlasl-video/videos"
if os.path.exists(dataset_dir):
    print("External drive is accessible!")
else:
    print("Error: Check if the drive is mounted correctly.")
