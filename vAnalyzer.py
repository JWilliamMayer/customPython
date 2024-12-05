import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json

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
    
    def analyze_motion(self, skip_frames=1, threshold=None):
        """
        Analyze motion in the video by calculating frame differences.
        
        :param skip_frames: Number of frames to skip between comparisons
        :param threshold: Optional threshold for motion detection
        :return: List of motion intensity for each compared frame pair
        """
        motion_intensities = []
        high_motion_frames = []
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
                
                # Check threshold if provided
                if threshold is not None and motion_intensity > threshold:
                    high_motion_frames.append({
                        'frame_number': len(motion_intensities),
                        'intensity': motion_intensity
                    })
            
            prev_frame = gray_frame
            
            # Skip frames if specified
            for _ in range(skip_frames):
                self.cap.read()
        
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return {
            'intensities': motion_intensities,
            'high_motion_frames': high_motion_frames
        }
    
    def detect_color_distribution(self, color_space='BGR'):
        """
        Analyze color distribution across the video.
        
        :param color_space: Color space to analyze (BGR, HSV, LAB)
        :return: Dictionary of average color channel intensities
        """
        channel_avgs = {
            'BGR': ['blue_channel', 'green_channel', 'red_channel'],
            'HSV': ['hue', 'saturation', 'value'],
            'LAB': ['L', 'A', 'B']
        }
        
        if color_space not in channel_avgs:
            raise ValueError(f"Unsupported color space. Choose from {list(channel_avgs.keys())}")
        
        channel_data = [[] for _ in range(3)]
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert color space if needed
            if color_space == 'HSV':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            elif color_space == 'LAB':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Calculate average channel intensities
            for i in range(3):
                channel_data[i].append(np.mean(frame[:,:,i]))
        
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Create return dictionary
        return dict(zip(
            channel_avgs[color_space], 
            [np.mean(channel) for channel in channel_data]
        ))
    
    def extract_frames(self, num_frames=10, output_dir=None):
        """
        Extract a specified number of frames from the video.
        
        :param num_frames: Number of frames to extract
        :param output_dir: Directory to save extracted frames
        :return: List of extracted frame paths
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(self.video_path), 'extracted_frames')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate frame intervals
        frame_step = max(1, self.total_frames // num_frames)
        
        extracted_frames = []
        for i in range(0, self.total_frames, frame_step):
            # Set the frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            
            # Read the frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Generate output filename
            frame_filename = os.path.join(output_dir, f'frame_{i:04d}.jpg')
            
            # Save the frame
            cv2.imwrite(frame_filename, frame)
            extracted_frames.append(frame_filename)
            
            # Stop if we've extracted enough frames
            if len(extracted_frames) >= num_frames:
                break
        
        # Reset video capture
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return extracted_frames
    
    def plot_motion_intensity(self, motion_data, output_path=None):
        """
        Create a plot of motion intensities throughout the video.
        
        :param motion_data: Dictionary containing motion intensities
        :param output_path: Optional path to save the plot
        """
        motion_intensities = motion_data['intensities']
        
        plt.figure(figsize=(12, 6))
        plt.plot(motion_intensities)
        plt.title('Motion Intensity Throughout Video')
        plt.xlabel('Frame Comparison')
        plt.ylabel('Motion Intensity')
        plt.tight_layout()
        
        # Determine output path
        if output_path is None:
            output_path = f"{os.path.splitext(self.video_path)[0]}_motion_intensity.png"
        
        plt.savefig(output_path)
        print(f"Motion intensity plot saved to: {output_path}")
        plt.close()
        
        # Optionally print high motion frames
        if motion_data['high_motion_frames']:
            print("\nHigh Motion Frames:")
            for frame in motion_data['high_motion_frames']:
                print(f"Frame {frame['frame_number']}: Intensity {frame['intensity']:.2f}")
    
    def plot_color_distribution(self, color_distribution, output_path=None):
        """
        Create a bar plot of color channel averages.
        
        :param color_distribution: Dictionary of color channel averages
        :param output_path: Optional path to save the plot
        """
        plt.figure(figsize=(8, 5))
        plt.bar(color_distribution.keys(), color_distribution.values())
        plt.title('Average Color Channel Intensities')
        plt.ylabel('Intensity')
        plt.tight_layout()
        
        # Determine output path
        if output_path is None:
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
    parser = argparse.ArgumentParser(description='Advanced Video Analysis Tool')
    
    # Input and output arguments
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output-dir', type=str, help='Directory to save output files')
    
    # Motion analysis arguments
    parser.add_argument('--skip-frames', type=int, default=5, 
                        help='Number of frames to skip between comparisons (default: 5)')
    parser.add_argument('--motion-threshold', type=float, 
                        help='Threshold for detecting high motion frames')
    
    # Color analysis arguments
    parser.add_argument('--color-space', type=str, choices=['BGR', 'HSV', 'LAB'], 
                        default='BGR', help='Color space for analysis (default: BGR)')
    
    # Frame extraction arguments
    parser.add_argument('--extract-frames', type=int, 
                        help='Number of frames to extract from the video')
    
    # Detailed output arguments
    parser.add_argument('--metadata', action='store_true', 
                        help='Print detailed video metadata')
    parser.add_argument('--output-format', choices=['text', 'json'], 
                        default='text', help='Format for output (default: text)')
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Create video analyzer
        analyzer = VideoAnalyzer(args.video_path)
        
        # Prepare output directory
        output_dir = args.output_dir or os.path.dirname(args.video_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect and process results
        results = {}
        
        # Get and print metadata
        metadata = analyzer.get_video_metadata()
        results['metadata'] = metadata
        
        if args.metadata:
            print("Video Metadata:")
            if args.output_format == 'json':
                print(json.dumps(metadata, indent=2))
            else:
                for key, value in metadata.items():
                    print(f"{key}: {value}")
        
        # Analyze motion
        motion_data = analyzer.analyze_motion(
            skip_frames=args.skip_frames, 
            threshold=args.motion_threshold
        )
        results['motion_analysis'] = motion_data
        
        # Plot motion intensity
        motion_plot_path = os.path.join(output_dir, 'motion_intensity.png')
        analyzer.plot_motion_intensity(motion_data, output_path=motion_plot_path)
        
        # Analyze color distribution
        color_distribution = analyzer.detect_color_distribution(
            color_space=args.color_space
        )
        results['color_distribution'] = color_distribution
        
        # Plot color distribution
        color_plot_path = os.path.join(output_dir, 'color_distribution.png')
        analyzer.plot_color_distribution(color_distribution, output_path=color_plot_path)
        
        # Extract frames if requested
        if args.extract_frames:
            extracted_frames = analyzer.extract_frames(
                num_frames=args.extract_frames, 
                output_dir=output_dir
            )
            results['extracted_frames'] = extracted_frames
        
        # Close the video capture
        analyzer.close()
        
        # Output results in specified format
        if args.output_format == 'json':
            output_file = os.path.join(output_dir, 'video_analysis.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()