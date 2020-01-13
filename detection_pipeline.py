import cv2
import numpy as np

import utils


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
        cv2.imshow('fg_mask', fg_mask)
        
        return context

class VehicleCounter(PipelineProcessor):
    '''
        Count the number of vehicles entering the exit zones.

        For this vehicle detection will find the centeroids of 
        detected vehicles and we will use that information to count
        the number of vehicles entering the exit zones.
    '''

    def __init__(self, exit_masks=[], path_size=10, max_dist_lanes=10):
        super(VehicleCounter, self).__init__()
        self.vehicle_count = 0
        self.exit_masks = exit_masks
        self.path_size = path_size
        self.max_dist_lanes = max_dist_lanes

        self.paths = []

    def check_vehicle_exit(self, centroid):
        for exit_mask in self.exit_masks:
            try:
                if exit_mask[centroid[1]][centroid[0]] == 255:
                    return True
            except:
                return True
        return False

    def __call__(self, context):
        context['exit_masks'] = self.exit_masks
        context['vehicle_count'] = self.vehicle_count
        context['paths'] = self.paths

        vehicles = context['vehicles']
        if not vehicles:
            return context

        centroids = np.array(vehicles)[:,1]
        centroids = centroids.tolist()
        
        if not self.paths:
            for centroid in centroids:
                self.paths.append([centroid])
        else:
            '''
                If the centroid is not empty keep only path_size centroids.
                use the distance to find the vehicles crossing the exit zones.
                
                From all the centroids the one that matches the existing path
                is the one that is closest to the previous path point on x-axis.
            '''
            new_paths = []
            
            for path in self.paths:
                min_dist_x_axis = 99999999
                match = None
                for centroid in centroids:
                    if len(path) == 1:
                        d = utils.distance(centroid, path[-1])
                    else:
                        predicted_next_x = 2 * path[-1][0] - path[-2][0]
                        predicted_next_y = 2 * path[-1][1] - path[-2][1]
                        d = utils.distance(centroids[0], (predicted_next_x, predicted_next_y)) 

                    if d < min_dist_x_axis:
                        min_dist_x_axis = d
                        match = centroid
                
                if match and min_dist_x_axis <= self.max_dist_lanes:
                    centroids.remove(centroid)
                    path.append(centroid)
                    new_paths.append(path)
                
                if not match:
                    new_paths.append(path)
            self.paths = new_paths

            # Add the points that had min_dist_x_axis > self.max_dist_lanes
            # and which have not crossed the exit zones
            for centroid in centroids:
                if not self.check_vehicle_exit(centroid[0]):
                    self.paths.append([centroid])
            
            for i, path in enumerate(self.paths):
                if len(path) > self.path_size:
                    self.paths[i] = path[:self.path_size]

            # count vehicles and drop counted pathes:
            new_paths = []
            for i, path in enumerate(self.paths):
                print(self.paths)
                d = path[-2:]
                # need at list two points to count, last point should be in 
                # exit zone, second last point should not be in exist zone. 
                if (
                    len(d) >= 2
                    and not self.check_vehicle_exit(d[0]) 
                    and self.check_vehicle_exit(d[1])
                ):
                    self.vehicle_count += 1
                    print(d)
                else:
                    # prevent linking with path that already in exit zone
                    add = True
                    # for p in path:
                        # if self.check_vehicle_exit(p[1]):
                        #     add = False
                        #     print('nooo')
                        #     break
                    if add:
                        new_paths.append(path)

            self.paths = new_paths

        context['vehicle_count'] = self.vehicle_count
        context['paths'] = self.paths
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
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), cv2.FILLED)
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