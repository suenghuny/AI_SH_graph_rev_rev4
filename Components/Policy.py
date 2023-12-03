import numpy as np
from utils import *

class Policy:

    def __init__(self, env, rule = 'rule2', temperatures = [10, 20]):
        self.env = env
        self.rule = rule
        self.temperature1 = temperatures[0]
        self.temperature2 = temperatures[1]


    def get_action(self, avail_action_list, target_distance_list, air_alert, open_fire_distance = 5):
        if self.rule == 'rule1':
            actions = list()
            for idx in range(len(avail_action_list)):
                avail_action = np.array(avail_action_list[idx])
                avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                actions.append(
                    np.random.choice(avail_actions_index, p=softmax(target_distance_list[idx], temperature=0)))

        if self.rule == 'rule2':
            actions = list()
            #print(target_distance_list)
            if air_alert == True:

                #print(self.temperature1, target_distance_list[0], softmax(target_distance_list[0], temperature = self.temperature1))
                #print(self.temperature2, target_distance_list[0], softmax(target_distance_list[0], temperature = self.temperature2))

                for idx in range(len(avail_action_list)):
                    avail_action = np.array(avail_action_list[idx])
                    avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                    actions.append(np.random.choice(avail_actions_index, p = softmax(target_distance_list[idx], temperature = self.temperature1, debug = True)))
            else:
                for idx in range(len(avail_action_list)):
                    avail_action = np.array(avail_action_list[idx])
                    avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                    # print("ㅇㅇㅇㅇ", avail_actions_index)
                    # print("ㄴㄴㄴㄴㄴ", target_distance_list)
                    actions.append(np.random.choice(avail_actions_index, p = softmax(target_distance_list[idx], temperature = self.temperature2, debug = True)))
        if self.rule == 'rule3':
            actions = list()

            target_distance_list = [[d if d < open_fire_distance or
                                          target_distance_list[0].index(d) == 0
                                     else 1000000 for d in target_distance_list[0] ]]
            #print(target_distance_list)
            if air_alert == True:

                #print(self.temperature1, target_distance_list[0], softmax(target_distance_list[0], temperature = self.temperature1))
                #print(self.temperature2, target_distance_list[0], softmax(target_distance_list[0], temperature = self.temperature2))

                for idx in range(len(avail_action_list)):
                    avail_action = np.array(avail_action_list[idx])
                    avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                    actions.append(np.random.choice(avail_actions_index, p = softmax(target_distance_list[idx], temperature = self.temperature1, debug = True)))
            else:
                for idx in range(len(avail_action_list)):
                    avail_action = np.array(avail_action_list[idx])
                    avail_actions_index = np.array(np.where(avail_action == True)).reshape(-1)
                    actions.append(np.random.choice(avail_actions_index, p = softmax(target_distance_list[idx], temperature = self.temperature2, debug = True)))
        return actions



