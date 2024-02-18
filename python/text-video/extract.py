import os

def list_video_names(folder_path):
    video_names = []
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.mp4'):
                video_names.append(file_name)
    return video_names

def main():
    folder_path = r"C:\SIH1344_19564_INNOVISIONERS\source_code\Python part\khacks!\video"  # Change this to your folder path
    video_names = list_video_names(folder_path)
    if video_names:
        print("Video names in folder:")
        for name in video_names:
            print(name)
    else:
        print("No video files found in the folder.")

if __name__ == "__main__":
    main()