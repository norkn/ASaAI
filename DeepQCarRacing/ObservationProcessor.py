#import cv2
import numpy as np

class ObservationProcessor:

    @staticmethod
    def _threshold_and_sum(cropped_state):
        thresholded_state = np.where(cropped_state > 0, 1, 0)
        return np.sum(thresholded_state)

    @staticmethod
    def _check_vision(vision_array):
        for i in vision_array[0]:
            if i[0] != i[1] or i[1] != i[2]:
                return 0
        return 1

    @staticmethod
    def get_speed(observation):
        gray_state = observation[:, :, 0]#cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[88:93, 12:13]
        #observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def get_left_steering(observation):
        gray_state = observation[:, :, 0]#cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[89:90, 41:47]
        #observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def get_right_steering(observation):
        gray_state = observation[:, :, 0]#cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        cropped_state = gray_state[89:90, 48:54]
        #observation_resized = cv2.resize(cropped_state, (40 * cropped_state.shape[1], 40 * cropped_state.shape[0]))
        return ObservationProcessor._threshold_and_sum(cropped_state)

    @staticmethod
    def get_vision(observation):
        near_vision_left = observation[66:67, 44:45]
       
        near_vision_right = observation[66:67, 51:52]
        far_vision_left = observation[46:47, 44:45]
        far_vision_right = observation[46:47, 51:52]
        vision_stripe = observation[50:51, 38:58]
        #vision_right = observation[70:71, 59:60]


        on_grass_left = observation[70:71, 46:47]
        on_grass_right = observation[70:71, 49:50]

        near_vision_left = ObservationProcessor._check_vision(near_vision_left)
        near_vision_right = ObservationProcessor._check_vision(near_vision_right)
        far_vision_left = ObservationProcessor._check_vision(far_vision_left)
        far_vision_right = ObservationProcessor._check_vision(far_vision_right)
        vision_stripe = ObservationProcessor._check_vision(vision_stripe)
        #vision_right = ObservationProcessor._check_vision(vision_right)

        on_grass_left = 0 if ObservationProcessor._check_vision(on_grass_left) else 1
        on_grass_right = 0 if ObservationProcessor._check_vision(on_grass_right) else 1

        is_on_grass = 1 if on_grass_left == 1 and on_grass_right == 1 else 0

        return [near_vision_left, near_vision_right, far_vision_left, far_vision_right, vision_stripe, is_on_grass]

    @staticmethod
    def get_state(observation):
        speed = ObservationProcessor.get_speed(observation)
        left_steering = ObservationProcessor.get_left_steering(observation)
        right_steering = ObservationProcessor.get_right_steering(observation)
        vision = ObservationProcessor.get_vision(observation)
        return [speed, left_steering, right_steering] + list(vision)