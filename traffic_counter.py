import cv2
import numpy as np

from detection_pipeline import (PipelineRunner, VehicleDetection, Visualizer)


#=================================================================
VIDEO_SOURCE = 'traffic.mp4'
EXIT_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],     # right side
    [[0, 400], [645, 400], [645, 0], [0, 0]]                # left side
])
SHAPE_VIDEO = (720,1280)
#=================================================================



def main():

    # create exit masks for the exit parts for counting vehicles.
    base = np.zeros(SHAPE_VIDEO + (3,), dtype='uint8')
    exit_masks = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]
    print(exit_masks)

    bg_subtractor = cv2.createBackgroundSubtractorKNN()

    # Create a pipeline for processing the frames.
    pipeline = PipelineRunner([
        VehicleDetection(bg_subtractor),
        Visualizer()
    ])

    capture = cv2.VideoCapture(VIDEO_SOURCE)

    _frame_number = -1      # real frame number 
    frame_number = -1       # frame number sent to the context
    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            _frame_number += 1
            # Skip every second frame.
            if _frame_number % 2 != 0:
                continue

            frame_number += 1
            
            pipeline.set_context({
                'frame': frame,
                'frame_number': frame_number,
                'exit_masks': [exit_masks],
                'vehicle_count': 0
            })
            
            pipeline.run()

            key = cv2.waitKey(100)
            if key == ord('q'):
                break


#=================================================================

if __name__ == "__main__":
    main()