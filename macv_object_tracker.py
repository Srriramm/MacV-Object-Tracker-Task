import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict


class TrailTracker:
    def __init__(self, max_trail_length=30):
        self.trails = defaultdict(list)
        self.max_trail_length = max_trail_length
        self.colors = {}
        # Add time tracking for each object
        self.object_times = defaultdict(int)
        self.first_appearances = defaultdict(int)

    def generate_color(self, track_id):
        if track_id not in self.colors:
            hue = np.random.random()
            saturation = 0.9
            value = 0.9
            rgb_color = tuple(round(i * 255) for i in cv2.cvtColor(
                np.uint8([[[hue * 179, saturation * 255, value * 255]]]),
                cv2.COLOR_HSV2BGR)[0][0])
            self.colors[track_id] = rgb_color
        return self.colors[track_id]

    def update_trail(self, track_id, centroid):
        self.trails[track_id].append(centroid)
        if len(self.trails[track_id]) > self.max_trail_length:
            self.trails[track_id].pop(0)

    def draw_trail(self, frame, track_id):
        if track_id in self.trails and len(self.trails[track_id]) > 1:
            color = self.generate_color(track_id)
            points = np.array(self.trails[track_id], dtype=np.int32)

            for i in range(1, len(points)):
                alpha = (i / len(points)) * 0.8 + 0.2
                current_color = tuple(int(c * alpha) for c in color)
                pt1 = tuple(points[i - 1])
                pt2 = tuple(points[i])
                cv2.line(frame, pt1, pt2, current_color, 2)


def format_time_ms(frame_count, fps):
    """Convert frame count to seconds and milliseconds format"""
    total_seconds = frame_count / fps
    seconds = int(total_seconds)
    milliseconds = int((total_seconds - seconds) * 1000)
    return f"{seconds}.{milliseconds:03d}s"


def track_unique_objects(video_path, model_path="yolov8n.pt"):
    # Load the YOLO model
    model = YOLO(model_path)

    # Initialize DeepSort tracker
    tracker = DeepSort(
        max_age=50,
        n_init=3,
        nn_budget=70,
        max_cosine_distance=0.4,
        embedder_gpu=True
    )

    # Initialize trail tracker
    trail_tracker = TrailTracker(max_trail_length=30)

    # Open the video source
    video_source = cv2.VideoCapture(video_path)
    if not video_source.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video properties
    frame_width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Using the specified FPS

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    # Dictionary to track unique IDs for each class
    unique_ids_by_class = {}
    frame_count = 0

    while True:
        ret, frame = video_source.read()
        if not ret:
            print("End of video or failed to read frame.")
            break

        frame_count += 1
        current_time = format_time_ms(frame_count, fps)

        # Run YOLO inference
        results = model(frame)

        detections = []
        for result in results:
            for *bbox, conf, class_id in result.boxes.data:
                class_name = model.names[int(class_id)]
                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, bbox)
                    width = x2 - x1
                    height = y2 - y1
                    detections.append(([x1, y1, width, height], conf, int(class_id)))

        # Update tracks
        tracks = tracker.update_tracks(detections, frame=frame)

        # Process and draw tracks
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            class_id = track.det_class
            class_name = model.names[class_id]

            # Update time tracking
            if track_id not in trail_tracker.first_appearances:
                trail_tracker.first_appearances[track_id] = frame_count
            trail_tracker.object_times[track_id] = frame_count - trail_tracker.first_appearances[track_id]

            # Calculate centroid
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Update and draw trail
            trail_tracker.update_trail(track_id, centroid)
            color = trail_tracker.generate_color(track_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw centroid
            cv2.circle(frame, centroid, 4, color, -1)

            # Draw trail
            trail_tracker.draw_trail(frame, track_id)

            # Calculate time for this object
            object_time = format_time_ms(trail_tracker.object_times[track_id], fps)

            # Draw label with time
            label = f"{class_name} ID: {track_id} Time: {object_time}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Update unique IDs tracking
            if class_name not in unique_ids_by_class:
                unique_ids_by_class[class_name] = set()
            unique_ids_by_class[class_name].add(track_id)

        # Display object counts and current video time
        y_offset = 30
        cv2.putText(frame, f"Video Time: {current_time}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 20

        for class_name, ids in unique_ids_by_class.items():
            text = f"{class_name.capitalize()}: {len(ids)}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 20

        # Write frame to output video
        out.write(frame)

        # Display frame
        cv2.imshow("Unique Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print final statistics with time information
    print("\nFinal statistics:")
    for class_name, ids in unique_ids_by_class.items():
        print(f"\n{class_name.capitalize()}:")
        print(f"Total unique objects: {len(ids)}")
        for track_id in ids:
            object_time = format_time_ms(trail_tracker.object_times[track_id], fps)
            print(f"ID {track_id}: Duration {object_time}")

    # Release resources
    video_source.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "macv-obj-tracking-video.mp4"  # Replace with "0" for webcam
    track_unique_objects(video_path)