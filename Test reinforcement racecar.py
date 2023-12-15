import gymnasium as gym
import cv2
import numpy as np


env = gym.make("CarRacing-v2", render_mode = "human")
#observation, info = env.reset()

# Anzahl der Zustände und Aktionen
n_states = 4096  # Beispiel: 5 Zustände
n_actions = 5  # Beispiel: 2 Aktionen

# Initialisieren der Q-Tabelle
Q_table = np.zeros((n_states, n_actions))

# Parameter
learning_rate = 0.1
discount_factor = 0.99
num_episodes = 1000  # Anzahl der Episoden für das Training


# Zustand diskretisieren (hypothetische Funktion)
def discretize_state(state):
    # Diese Funktion muss basierend auf Ihrer Zustandsdefiniton implementiert werden
    return int(state)

for episode in range(num_episodes):
    state = env.reset()
    state = discretize_state(state)
    done = False

    while not done:

        action = env.action_space.sample(n_actions)  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

#        auto_crop = cv2.rectangle(observation, (40, 60), (55,76), (0,126,0), 1)
#        car_front_left = cv2.rectangle(auto_crop, (42, 48), (44,50), (0,255, 255), 1)
#        car_front_rigt = cv2.rectangle(car_front_left, (51, 48), (53,50), (0,255, 255), 1)
#        car_back_left = cv2.rectangle(car_front_rigt, (42, 65), (44,67), (0,255, 255), 1)
#        car_back_rigt = cv2.rectangle(car_back_left, (51, 65), (53,67), (0,255, 255), 1)
#        left_crop = cv2.rectangle(car_back_rigt,(44,87), (47, 89), (0, 255, 255), 1)
#        right_crop = cv2.rectangle(left_crop,(48,87), (51,89), (0, 255, 255), 1)
#        speed_crop = cv2.rectangle(right_crop, (12,90), (14,93),(0, 255, 255),1)

        left_steering = observation[88:89, 45:49]  #[Höhe, Breit]
        gray_left_steering = cv2.cvtColor(left_steering, cv2.COLOR_BGR2GRAY)
        retcode1, binary_left_steering = cv2.threshold(gray_left_steering, 3, 1, cv2.THRESH_BINARY)
        sum_bin_l_steer = np.sum(binary_left_steering) + 1

        right_steering = observation[88:89, 49:53]  #[Höhe, Breit]
        gray_right_steering = cv2.cvtColor(right_steering, cv2.COLOR_BGR2GRAY)
        retcode2, binary_right_steering = cv2.threshold(gray_right_steering, 3, 1, cv2.THRESH_BINARY)
        sum_bin_r_steer = np.sum(binary_right_steering) + 1

        speed_steering = observation[90:94, 13:14]  #[Höhe, Breit]
        gray_speed_steering = cv2.cvtColor(speed_steering, cv2.COLOR_BGR2GRAY)
        retcode3, binary_speed_steering = cv2.threshold(gray_speed_steering, 3, 1, cv2.THRESH_BINARY)
        sum_bin_speed = np.sum(binary_speed_steering) + 1

        front_left = observation[49:50, 43:44]  #[Höhe, Breit]
        gray_front_left = cv2.cvtColor(front_left, cv2.COLOR_BGR2GRAY)
        retcode4, front_left_pixel = cv2.threshold(gray_front_left, 120, 1, cv2.THRESH_BINARY)
        f_l_p = np.sum(front_left_pixel) + 1

        front_right = observation[49:50, 52:53]  # [Höhe, Breit]
        gray_front_right = cv2.cvtColor(front_right, cv2.COLOR_BGR2GRAY)
        retcode5, front_right_pixel = cv2.threshold(gray_front_right, 120, 1, cv2.THRESH_BINARY)
        f_r_p = np.sum(front_right_pixel) + 1

        back_left = observation[66:67, 43:44]  # [Höhe, Breit]
        gray_back_left = cv2.cvtColor(back_left, cv2.COLOR_BGR2GRAY)
        retcode6, back_left_pixel = cv2.threshold(gray_back_left, 120, 1, cv2.THRESH_BINARY)
        b_l_p = np.sum(back_left_pixel) + 1

        back_right = observation[66:67, 52:53]  # [Höhe, Breit]
        gray_back_right = cv2.cvtColor(back_right, cv2.COLOR_BGR2GRAY)
        retcode7, back_right_pixel = cv2.threshold(gray_back_right, 120, 1, cv2.THRESH_BINARY)
        b_r_p = np.sum(back_right_pixel) + 1

        observ_state = (sum_bin_r_steer * sum_bin_l_steer * sum_bin_speed *
                        ((b_r_p * (b_r_p + b_l_p + f_r_p + f_l_p)) + (b_l_p * (b_r_p + b_l_p + f_r_p + f_l_p)) + (f_r_p * (b_r_p + b_l_p + f_r_p + f_l_p)) + (f_l_p * (b_r_p + b_l_p + f_r_p + f_l_p))))

        next_state = discretize_state(observ_state)

        old_value = Q_table[current_state, action]
        next_max = np.max(Q_table[next_state])

        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        Q_table[current_state, action] = new_value

        current_state = next_state


    #    show = cv2.resize(speed_crop, (1000, 900), interpolation= cv2.INTER_LINEAR)
    #    cv2.imshow("Capture", speed_crop)
    #    cv2.imshow("Crop", show)
    #    print(sum_bin_speed, "    ", sum_bin_l_steer, "    ", sum_bin_r_steer,"     ", observ_array)


#        if terminated or truncated:
#            observation, info = env.reset()

print(Q_table)
env.close()
