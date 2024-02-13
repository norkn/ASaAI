import gymnasium as gym
import cv2
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("CarRacing-v2", render_mode = "human", continuous = False)
observation, info = env.reset()

# Anzahl der Zustände und Aktionen
n_states = 4096  # Beispiel: 5 Zustände
n_actions = 5  # 5 aktionen die er gleichzeitig ausführen kann

# Initialisieren der Q-Tabelle
Q_table = np.random.uniform(low= 0, high= 0, size=(n_states, n_actions))

ep_rewards = [] #Dictionary
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []} #Dictionary welches die rewards pro episode abspeichert.

# Parameter
learning_rate = 0.1
discount_factor = 0.95
num_episodes = 10000 # Anzahl der Episoden für das Training (24000)

epsilon = 0.95
start_y_decaying = 1
end_y_decaying = num_episodes // 2

epsilon_decay_value = epsilon/(end_y_decaying - start_y_decaying)

show_every = 1000  #zeig mir alle 2000 episoden, das der Agent noch am abreiten ist.

current_state = 1
next_state = 1

done = False

for episode in range(num_episodes):
#    print(epsilon)
    episode_reward = 0 #für das Dictionary

    if episode % show_every == 0:
        print(episode)
        render = True
    else:
        render = False

    if render:
        env.reset()

    if np.random.random() > epsilon:  #np.random.random() sind random zahlen, die alle kleiner als 1 sind. Zu Beginn von epsilon ist epsilon also größer als np.random.random() und wir mit zunehmender Wharscheinlichkeit kleiner als np.random...
        action = np.argmax(Q_table[current_state]) #füttere die Action mit den momentanwerten aus der Qtabelle des states. daraus wird dann action, und action bestimmt den nächsten state (observation)
    else:
        action = np.random.randint(0, 5)

    observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward #für das Dictionary

    a = 1
    b = 1
    c = 1
    d = 1
    e = 1
    f = 1
    g = 1
    h = 1
    i = 1
#    print(current_state)
#    print(action)
#    print(Q_table[current_state])

    class PixelObserv:

        def __init__(self, sls = 0, srs = 0, sss = 0, flp = 0, frp = 0, blp = 0, brp = 0, nwp= 0):
            self.pixelvalue = 1
            self.newpixel = int(nwp)
            self.sumleftsteering = 1
            self.newsls = int(sls)
            self.sumrightsteering = 1
            self.newsrs = int(srs)
            self.sumspeed = 1
            self.newsss = int(sss)
            self.f_l_p = 1
            self.newflp = int(flp)
            self.f_r_p = 1
            self.newfrp = int(frp)
            self.b_l_p = 1
            self.newblp = int(blp)
            self.b_r_p = 1
            self.newbrp = int(brp)
            self.reward = 1
            self.action = 1

        def getobeservation(self):
            self.observation, self.reward, self.terminated, self.truncated, self.info = env.step(action)
            return self.reward

        def getleftsteering(self):
            left_steering = observation[88:89, 45:49]  # [Höhe, Breit]
            gray_left_steering = cv2.cvtColor(left_steering, cv2.COLOR_BGR2GRAY)
            retcode1, binary_left_steering = cv2.threshold(gray_left_steering, 3, 1, cv2.THRESH_BINARY)
            self.sumleftsteering = np.sum(binary_left_steering) + 1
            self.newsls = self.sumleftsteering
            return self.newsls

        def getrightsteering(self):
            right_steering = observation[88:89, 49:53]  # [Höhe, Breit]
            gray_right_steering = cv2.cvtColor(right_steering, cv2.COLOR_BGR2GRAY)
            retcode2, binary_right_steering = cv2.threshold(gray_right_steering, 3, 1, cv2.THRESH_BINARY)
            self.sumrightsteering = np.sum(binary_right_steering) + 1
            self.newsrs = self.sumrightsteering
            return self.newsrs

        def getspeedsteering(self):
            speed_steering = observation[90:94, 13:14]  # [Höhe, Breit]
            gray_speed_steering = cv2.cvtColor(speed_steering, cv2.COLOR_BGR2GRAY)
            retcode3, binary_speed_steering = cv2.threshold(gray_speed_steering, 3, 1, cv2.THRESH_BINARY)
            self.sumspeed = np.sum(binary_speed_steering) + 1
            self.newsss = self.sumspeed
            return self.newsss

        def getpixel_f_l(self):
            front_left = observation[49:50, 43:44]  # [Höhe, Breit]
            gray_front_left = cv2.cvtColor(front_left, cv2.COLOR_BGR2GRAY)
            retcode4, front_left_pixel = cv2.threshold(gray_front_left, 120, 1, cv2.THRESH_BINARY)
            self.f_l_p = np.sum(front_left_pixel) + 1
            self.newflp = self.f_l_p
            return self.newflp

        def getpixel_f_r(self):
            front_right = observation[49:50, 52:53]  # [Höhe, Breit]
            gray_front_right = cv2.cvtColor(front_right, cv2.COLOR_BGR2GRAY)
            retcode5, front_right_pixel = cv2.threshold(gray_front_right, 120, 1, cv2.THRESH_BINARY)
            self.f_r_p = np.sum(front_right_pixel) + 1
            self.newfrp = self.f_r_p
            return self.newfrp

        def getpixelb_l(self):
            back_left = observation[66:67, 43:44]  # [Höhe, Breit]
            gray_back_left = cv2.cvtColor(back_left, cv2.COLOR_BGR2GRAY)
            retcode6, back_left_pixel = cv2.threshold(gray_back_left, 120, 1, cv2.THRESH_BINARY)
            self.b_l_p = np.sum(back_left_pixel) + 1
            self.newblp = self.b_l_p
            return self.newblp

        def getpixelb_r(self):
            back_right = observation[66:67, 52:53]  # [Höhe, Breit]
            gray_back_right = cv2.cvtColor(back_right, cv2.COLOR_BGR2GRAY)
            retcode7, back_right_pixel = cv2.threshold(gray_back_right, 120, 1, cv2.THRESH_BINARY)
            self.b_r_p = np.sum(back_right_pixel) + 1
            self.newbrp = self.b_r_p
            return self.newbrp

        def getobservestate(self):
            self.pixelvalue = (self.newsls * self.newsrs * self.newsss *
                            ((self.newbrp * (self.newbrp + self.newblp + self.newfrp + self.newflp)) +
                             (self.newblp * (self.newbrp + self.newblp + self.newfrp + self.newflp)) +
                             (self.newfrp * (self.newbrp + self.newblp + self.newfrp + self.newflp)) +
                             (self.newflp * (self.newbrp + self.newblp + self.newfrp + self.newflp))))
            self.newpixel = self.pixelvalue
            return self.newpixel

    PixelObserv1 = PixelObserv(a, b, c, d, e, f, g, h)

    a = PixelObserv1.getleftsteering()
    b = PixelObserv1.getrightsteering()
    c = PixelObserv1.getspeedsteering()
    d = PixelObserv1.getpixel_f_l()
    e = PixelObserv1.getpixel_f_r()
    f = PixelObserv1.getpixelb_l()
    g = PixelObserv1.getpixelb_r()
    h = PixelObserv1.getobservestate()

#    print(current_state, next_state)
    if not done:
        #Q-Value Berechnung
        next_state = PixelObserv1.getobservestate()

        old_value = Q_table[current_state, action]
        next_max = np.max(Q_table[next_state])

        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        Q_table[current_state, action] = new_value

    elif next_state[0] >= env.goal_position:
        print(f"We made it on episode{episode}")
        Q_table[current_state + action] = 0

    current_state = next_state

    if end_y_decaying >= episode  >= start_y_decaying:
        epsilon -=epsilon_decay_value

    ep_rewards.append(episode_reward) #für das Dictionary

    if not episode % show_every: #für das Dictionary ---- wenn episode modulo ist gleich 0 dann:
        average_reward = sum(ep_rewards[-show_every:])/len(ep_rewards[-show_every:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-show_every:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-show_every:]))

        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-show_every:])} max: {max(ep_rewards[-show_every:])}")

print(Q_table)
print("OldV:\n", old_value)
print("NewV\n", new_value)
print(aggr_ep_rewards['avg'])
env.close()

#Darstellung des Dictionaries mit Pyplot. Es werden die Rewards pro episode gezeigt, woran fortschritte und lernraten abgeleitet werden können.
# !!!!ACHTUNG!!! Es müssen mindestens 2 resets (show_every) durchlaufen werden damit ein array ensteht, welches geplottet werden kann!!
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.xlabel("episodes")
plt.ylabel("reward")
plt.legend(loc=4)
plt.show()

'''
Der Plot zeigt uns den minamlen und maximalen reward, den der Agent pro abgeschlossenen reset (alle x episoden) erreicht hat.
Je näher der average an dem maximum dran ist, desto besser ist die learningrate. Je länger man den Agent trainiert, mit episoden,
desto besser werden die werte für den reward, für max average und min. 
Mit steigenden episoden (x-achse) sollte auch der reward für alle steigen (y-achse).
MAx = bester agent , Min = schlechtester agent.
'''
