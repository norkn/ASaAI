import numpy as np

class ObservationProcessor:

    @staticmethod
    def _threshold_and_sum(cropped_state):
        thresholded_state = np.where(cropped_state > 0, 1, 0)
        
        return np.sum(thresholded_state)

    @staticmethod
    def _is_on_road(pixel):
        return int(pixel[0] == pixel[1] and pixel[1] == pixel[2] and pixel[0] > 0)

    @staticmethod
    def _get_value_from_pixels(observation, dX, dY):
        cropped_state = observation[dX[0]:dX[1], dY[0]:dY[1], :]
        gray_state = (cropped_state[:, :, 0] + cropped_state[:, :, 1] + cropped_state[:, :, 2]) / 3.

        return ObservationProcessor._threshold_and_sum(gray_state)

    @staticmethod
    def get_speed(observation):
        return ObservationProcessor._get_value_from_pixels(observation, (88, 93), (12, 13))

    @staticmethod
    def get_left_steering(observation):
        return ObservationProcessor._get_value_from_pixels(observation, (89, 90), (41, 47))

    @staticmethod
    def get_right_steering(observation):
        return ObservationProcessor._get_value_from_pixels(observation, (89, 90), (48, 54))

    @staticmethod
    def get_vision(observation):
        near_vision_left = observation[66][44]       
        near_vision_right = observation[66][51]

        far_vision_left = observation[46][44]
        far_vision_right = observation[46][51]

        vision_stripe = observation[50][48]

        on_grass_left = observation[70][46]
        on_grass_right = observation[70][49]

        near_vision_left = ObservationProcessor._is_on_road(near_vision_left)
        near_vision_right = ObservationProcessor._is_on_road(near_vision_right)

        far_vision_left = ObservationProcessor._is_on_road(far_vision_left)
        far_vision_right = ObservationProcessor._is_on_road(far_vision_right)

        vision_stripe = ObservationProcessor._is_on_road(vision_stripe)

        on_grass_left = 1 - ObservationProcessor._is_on_road(on_grass_left)
        on_grass_right = 1 - ObservationProcessor._is_on_road(on_grass_right)

        is_on_grass = int(on_grass_left == 1 and on_grass_right == 1)
    
        # vision_array = []
        # vision_step = 6
        # for x in range(int(len(observation[0]) / vision_step)):
            # for y in range(int(len(observation) / vision_step)):
                # pixel = ObservationProcessor()._is_on_road(observation[y * vision_step][x * vision_step])
                # vision_array.append(pixel)

        return [near_vision_left, near_vision_right, far_vision_left, far_vision_right, vision_stripe, is_on_grass]# + vision_array

    @staticmethod
    def get_state(observation):

        speed = ObservationProcessor.get_speed(observation)
        left_steering = ObservationProcessor.get_left_steering(observation)
        right_steering = ObservationProcessor.get_right_steering(observation)

        vision = ObservationProcessor.get_vision(observation)

        return [speed, left_steering, right_steering] + list(vision)
