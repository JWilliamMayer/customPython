import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

class VideoAnalyzer:
    def __init__(self, video_path):
        """
        Initialize the video analyzer with the given video file path.
        
        :param video_path: Path to the input video file
        """
        self.video_path = video_path
        
        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video file.")
        
        # Video metadata
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def get_video_metadata(self):
        """
        Return basic metadata about the video.
        
        :return: Dictionary containing video metadata
        """
        return {
            'filename': os.path.basename(self.video_path),
            'total_frames': self.total_frames,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'duration': self.total_frames / self.fps
        }
    
    def analyze_motion(self, skip_frames=1):
        """
        Analyze motion in the video by calculating frame differences.
        
        :param skip_frames: Number of frames to skip between comparisons
        :return: List of motion intensity for each compared frame pair
        """
        motion_intensities = []
        prev_frame = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                frame_diff = cv2.absdiff(prev_frame, gray_frame)
                motion_intensity = np.mean(frame_diff)
                motion_intensities.append(motion_intensity)
            
            prev_frame = gray_frame
            
            # Skip frames if specified
            for _ in range(skip_frames):
                self.cap.read()
        
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return motion_intensities
    
    def detect_color_distribution(self):
        """
        Analyze color distribution across the video.
        
        :return: Dictionary of average color channel intensities
        """
        blue_avg, green_avg, red_avg = [], [], []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Calculate average color channel intensities
            blue_avg.append(np.mean(frame[:,:,0]))
            green_avg.append(np.mean(frame[:,:,1]))
            red_avg.append(np.mean(frame[:,:,2]))
        
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return {
            'blue_channel': np.mean(blue_avg),
            'green_channel': np.mean(green_avg),
            'red_channel': np.mean(red_avg)
        }
    
    def plot_motion_intensity(self, motion_intensities):
        """
        Create a plot of motion intensities throughout the video.
        
        :param motion_intensities: List of motion intensities
        """
        plt.figure(figsize=(12, 6))
        plt.plot(motion_intensities)
        plt.title('Motion Intensity Throughout Video')
        plt.xlabel('Frame Comparison')
        plt.ylabel('Motion Intensity')
        plt.tight_layout()
        
        # Save plot to file
        output_path = f"{os.path.splitext(self.video_path)[0]}_motion_intensity.png"
        plt.savefig(output_path)
        print(f"Motion intensity plot saved to: {output_path}")
        plt.close()
    
    def plot_color_distribution(self, color_distribution):
        """
        Create a bar plot of color channel averages.
        
        :param color_distribution: Dictionary of color channel averages
        """
        plt.figure(figsize=(8, 5))
        plt.bar(color_distribution.keys(), color_distribution.values())
        plt.title('Average Color Channel Intensities')
        plt.ylabel('Intensity')
        plt.tight_layout()
        
        # Save plot to file
        output_path = f"{os.path.splitext(self.video_path)[0]}_color_distribution.png"
        plt.savefig(output_path)
        print(f"Color distribution plot saved to: {output_path}")
        plt.close()
    
    def close(self):
        """
        Release the video capture object.
        """
        self.cap.release()

def parse_arguments():
    """
    Parse command-line arguments.
    
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Analyze video file characteristics.')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--skip-frames', type=int, default=5, 
                        help='Number of frames to skip between comparisons (default: 5)')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Create video analyzer
        analyzer = VideoAnalyzer(args.video_path)
        
        # Get video metadata
        metadata = analyzer.get_video_metadata()
        print("Video Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        # Analyze motion
        motion_intensities = analyzer.analyze_motion(skip_frames=args.skip_frames)
        analyzer.plot_motion_intensity(motion_intensities)
        
        # Analyze color distribution
        color_distribution = analyzer.detect_color_distribution()
        print("\nColor Distribution:")
        print(color_distribution)
        analyzer.plot_color_distribution(color_distribution)
        
        # Close the video capture
        analyzer.close()
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()