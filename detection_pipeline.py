import cv2
import numpy as np


# ============================================================================
DEFAULT_MIN_SIZE_CONTOUR = 25
VEHICLE_RECT_COLOUR = (255, 0, 0)
CENTROID_COLOUR = (0,0,255)
EXIT_MASK_COLOR = (66, 183, 42)
# ============================================================================

class PipelineRunner(object):
    '''
        Very simple pipline.
        Just run passed processors in order with passing context from one to 
        another.
        You can also set log level for processors.
    '''

    def __init__(self, pipeline=None):
        self.pipeline = pipeline or []
        self.context = {}

    def set_context(self, data):
        self.context = data

    def add(self, processor):
        if not isinstance(processor, PipelineProcessor):
            raise Exception(
                'Processor should be an isinstance of PipelineProcessor.')
        else:
            self.pipeline.append(processor)

    def remove(self, name):
        for i, p in enumerate(self.pipeline):
            if p.__class__.__name__ == name:
                del self.pipeline[i]
                return True
        return False

    def run(self):
        for p in self.pipeline:
            self.context = p(self.context)

        return self.context

class PipelineProcessor(object):
    '''
        Base class for all the processors.
    '''

class VehicleDetection(PipelineProcessor):
    '''
        Detect moving vehicles and generate bounding reactangle coordinates.
        Use the background subtraction and use then filter the object based on their size.
    '''

    def __init__(self, bg_subtractor, min_size_contour=DEFAULT_MIN_SIZE_CONTOUR):
        super(VehicleDetection, self).__init__()

        self.bg_subtractor = bg_subtractor
        self.min_size_contour = min_size_contour

    def reduce_noise(self, frame):
        '''
            Filters vehicles in the frames to reduce noise.
        '''

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        frame = cv2.erode(frame, kernel, iterations=3)
        frame = cv2.dilate(frame, kernel, iterations=1)

        return frame

    def detect_vehicles(self, fg_mask, min_size_contour):
        '''
            detect the vehicles by filtering out the smaller objects.
        '''
        vehicles = []
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        # Check if the contour has a valid size and then add it to the list else continue
        for contour in contours:
            (left, top, width, height) = cv2.boundingRect(contour)
            if width >= min_size_contour and height >= min_size_contour:
                cx = left + (width // 2)
                cy = top + (height // 2)
                vehicles.append( ((left, top, width, height), (cx, cy)) )
    
        return vehicles

    def __call__(self, context):
        frame = context['frame'].copy()
        fg_mask = self.bg_subtractor.apply(frame, None, 0.005)
        fg_mask = self.reduce_noise(fg_mask)

        context['vehicles'] = self.detect_vehicles(fg_mask, self.min_size_contour)
        context['fg_mask'] = fg_mask
        
        return context

class Visualizer(PipelineProcessor):

    def __init__(self):
        super(Visualizer, self).__init__()

    def check_vehicle_exit(self, center, exit_masks=[]):
        for exit_mask in exit_masks:
            if exit_mask[center[1]][center[0]] == 255:
                return True
        return False


    def draw_vehicle_rects(self, frame, vehicles, exit_masks ):
        '''
            Draw rectangles around the vehicles.
        '''        
        for vehicle in vehicles:
            rect, centroid = vehicle

            if not self.check_vehicle_exit(centroid, exit_masks):
                x, y, w, h = rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), VEHICLE_RECT_COLOUR, 1)
                cv2.circle(frame, centroid, 2, CENTROID_COLOUR, -1)

        return frame

    def draw_ui(self, frame, vehicle_count, exit_masks):
        
        # this just add green mask with opacity to the image.
        # create a mask with the mask color and and with the mask area.
        for exit_mask in exit_masks:
            _frame = np.zeros(frame.shape, frame.dtype)
            _frame[:, :] = EXIT_MASK_COLOR
            mask = cv2.bitwise_and(_frame, _frame, mask=exit_mask)
            cv2.addWeighted(mask, 1, frame, 1, 0, frame)

        # drawing top block with counts
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, ("Vehicles passed: {count} ".format(count=vehicle_count)), (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return frame

    def __call__(self, context):
        frame = context['frame'].copy()
        vehicles = context['vehicles']
        exit_masks = context['exit_masks']
        vehicle_count = context['vehicle_count']

        frame = self.draw_ui(frame, vehicle_count, exit_masks)
        frame = self.draw_vehicle_rects(frame, vehicles, exit_masks)
        cv2.imshow('frame', frame)

        return context