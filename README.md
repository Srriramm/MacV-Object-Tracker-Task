# MacV-Object-Tracker-Task

This project implements object tracking using YOLOv8 for object detection and DeepSort for tracking, along with a custom trail tracker to visualize object movements over time. The project outputs a video with unique object tracking information, including the object's ID, class, and the duration of its appearance in the video. Additionally, the results have been visualized in an HTML file with the output video.

## Features

- **Object Detection**: Utilizes YOLOv8 for real-time object detection.
- **Object Tracking**: Implements DeepSort for tracking detected objects across frames.
- **Object Trails**: Tracks and visualizes the movement of objects with trails.
- **Unique Object IDs**: Assigns unique IDs to detected objects and tracks them throughout the video.
- **Time Tracking**: Calculates and displays the duration of each object's appearance in the video.
- **Output Video**: Saves an output video with tracked objects, including bounding boxes, trails, and IDs.
- **HTML Visualization**: Displays the results with the output video in an HTML file for easy viewing and analysis.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- DeepSort Realtime
- NumPy
- HTML (for result visualization)

To install the required libraries, use the requirements.txt file


## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Srriramm/MacV-Object-Tracker-Task.git
    cd object-tracking-yolo-deepsort
    ```
    
2. Download the YOLOv8 model (`yolov8n.pt`) and place it in the same directory as the script.

   You can download the YOLOv8 model from the official repository:
   [YOLOv8 Models](https://github.com/ultralytics/ultralytics)

3. Ensure that the video file you want to process is available. If you want to use your webcam, replace the `video_path` with `"0"`.

## Usage

To run the object tracking script, use the following command:

```bash
python track_objects.py
```

This will process the video specified in the `video_path` variable. The script will display the video with tracked objects and save the output to `output.mp4`.

### Video Path

In the script, you can specify the video file path for tracking. Replace `"macv-obj-tracking-video.mp4"` with the path to your video.

```python
video_path = "macv-obj-tracking-video.mp4"  
```

## How It Works

1. **Object Detection**: The YOLOv8 model detects objects in each frame of the video.
2. **Tracking**: The DeepSort tracker assigns a unique ID to each detected object and keeps track of them over subsequent frames.
3. **Trail Visualization**: A trail is drawn behind each object to visualize its movement. The trail length is configurable (default is 30 frames).
4. **Time Tracking**: The time an object appears in the video is calculated and displayed on the output video.
5. **Output**: The video with all the information (bounding boxes, IDs, trails, and times) is saved as `output.mp4`.
6. **HTML Visualization**: The results are visualized in an HTML file, allowing for easy access and review of the output video along with the associated object tracking data.

## Output

The resulting video will show:

- Bounding boxes around detected objects.
- Trails representing the movement of tracked objects.
- The unique ID and appearance time of each tracked object.
- An overall count of tracked objects per class.

Additionally, an HTML file is generated that contains the output video along with the tracked object information. This HTML file can be opened in any browser for quick visualization.

## Final Statistics

At the end of the video, the script will print the total number of unique objects tracked for each class, along with the duration of their appearance in the video.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection model.
- [DeepSort](https://github.com/nwojke/deep_sort) for the tracking algorithm.
- [OpenCV](https://opencv.org/) for computer vision functionality.

