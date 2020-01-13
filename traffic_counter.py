import cv2
import numpy as np

from detection_pipeline import (
    PipelineRunner, 
    VehicleDetection, 
    VehicleCounter,
    Visualizer
)


#=================================================================
VIDEO_SOURCE = 'traffic.mp4'
TRAINING_VIDEO = 'training_video.mp4'
EXIT_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],     # right side
    [[0, 400], [645, 400], [645, 0], [0, 0]]                # left side
])
SHAPE_VIDEO = (720,1280)

# VIDEO_SOURCE = 'video.avi'
# TRAINING_VIDEO = 'video.avi'
# EXIT_PTS = np.array([
#     [[260,0], [320, 0], [320, 176], [260, 174]]
# ])
# SHAPE_VIDEO = (176,320)
#=================================================================

def train_model(bg_subtractor, iterations=100):

    capture = cv2.VideoCapture(TRAINING_VIDEO)
    while capture.isOpened() and iterations > 0:
        ret, frame = capture.read()
        if ret:
            bg_subtractor.apply(frame, None, 0.05)
            iterations -= 1

            key = cv2.waitKey(100)
            if key == ord('q'):
                break

    return bg_subtractor

def main():
    # create exit masks for the exit parts for counting vehicles.
    # exit mask is VIDEO_SHAPE * 3 (pixels).
    base = np.zeros(SHAPE_VIDEO + (3,), dtype='uint8')
    exit_masks = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    # Create background subtraction and add train it.
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=False)
    bg_subtractor = train_model(bg_subtractor, 100)

    # Create a pipeline for processing the frames.
    pipeline = PipelineRunner([
        VehicleDetection(bg_subtractor),
        VehicleCounter([exit_masks]),
        Visualizer()
    ])

    capture = cv2.VideoCapture(VIDEO_SOURCE)
    _frame_number = -1                      # real frame number 
    frame_number = -1                       # frame number sent to the context
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
            })
            
            pipeline.run()
            key = cv2.waitKey(100)
            if key == ord('q'):
                break

#=================================================================

if __name__ == "__main__":
    main()