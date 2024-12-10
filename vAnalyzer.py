import cv2
import os
import json

def get_video_metadata(video_path):
    """
    Extract basic metadata from a video file.
    
    :param video_path: Path to the input video file
    :return: Dictionary containing core video metadata
    """
    # Check if file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file.")
    
    try:
        # Extract metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        # Return core metadata
        return {
            'total_frames': total_frames,
            'fps': round(fps, 2),
            'resolution': f'{width}x{height}',
            'duration': round(duration, 2)
        }
    
    finally:
        # Always release the video capture
        cap.release()

def main():
    import sys
    
    # Check if video path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path>")
        sys.exit(1)
    
    try:
        # Get video path from command-line argument
        video_path = sys.argv[1]
        
        # Get and print metadata
        metadata = get_video_metadata(video_path)
        
        # Print metadata in a readable format
        print(f"Total Frames: {metadata['total_frames']}")
        print(f"Frames per Second: {metadata['fps']}")
        print(f"Resolution: {metadata['resolution']}")
        print(f"Duration: {metadata['duration']} seconds")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()