"""
@Author: Ishaan Roy
File contains: Network Environment
TODO:
----------------------------------------
=> Create the network environment that contains the following components:
    -> Central Resources:
        1. Total capacity
        2. Slices
        3. Coverage
        4. Slice ratios
    -> Users:
        1. User frequency
        2. Connected slice
=> Implement multi-agent RL for each slice
----------------------------------------
ChangeLog
----------------------------------------

----------------------------------------

"""
import numpy as np
import matplotlib.pyplot as plt

class Slice:
    def __init__(self, name, slice_ratio, band_limits):
        self.name = name
        self.ratio= slice_ratio
        self.allocate_capacity(slice_ratio)
        self.connected_users = 0
        self.total_requested_bw = 0
        self.band_limits = band_limits



    def allocate_capacity(self, total_capacity):
        slice_cap = self.ratio*total_capacity
        self.slice_capacity = slice_cap
        self.init_capacity = slice_cap

    def accept_user(self, requested_bandwith):
        if self.band_limits[0] <= requested_bandwith <= self.band_limits[1]:
            if requested_bandwith <= self.slice_capacity:
                print('Request Accepted')
                self.slice_capacity -= requested_bandwith
                self.connected_users += 1
                self.total_requested_bw += requested_bandwith
                return True
            else:
                return False
        else:
            print('Requested Denied: Outside Band Limits!')
            return None

    def setState(self, total_capacity, n_users):
        users_connected_ratio = self.connected_users/n_users
        requested_bw_ratio = self.total_requested_bw/total_capacity
        remaining_bw_ratio = self.slice_capacity/self.init_capacity
        return (users_connected_ratio, requested_bw_ratio, remaining_bw_ratio)




class Network:
    def __init__(self, total_capcity, slice_info,
                 action_factor, low_limit, high_limit):
        self.total_capcity = total_capcity
        self.slice_info = slice_info
        self.distribute_bandwidth(slice_info)
        self.action_factor = action_factor
        self.actions = {
                        'INC': +0.05*action_factor,
                        'DEC': -0.025*action_factor,
                        'NO_CHANGE': 0,

        }
        self.possibleActions = [key for key in self.actions]
        self.low_limit = low_limit
        self.high_limit = high_limit

    def distribute_bandwidth(self, slice_info):
        self.slices = []
        for slice_name in slice_info:
            slice_instance = Slice(slice_name, slice_info[slice_name][0],
                                   slice_info[slice_name][1])

            self.slices.append(slice_instance)

    def isTerminalState(self, state):
        state = state.reshape(3, 3)
        state = np.matrix(state).transpose()
        sum_matrix = state.sum(axis=1)
        return sum_matrix[0] >= 0.9

    def step(self, action, n_users=100):
        """
        :param action:
        :param state: [connected_fraction_sl1, connected_fraction_sl2, connected_fraction_sl3,
                       requested_frac_sl1, requested_frac_sl2, requested_frac_sl3,
                       remaining_bw_sl1, remaining_bw_sl2, remaining_bw_sl3]
        :return: state, action, reward, info_dict
        """
        users_requests = np.random.randint(self.low_limit, self.high_limit, size=n_users)
        reward_dict = {'eMBB': 0, 'mMTC': 0, 'URLLC': 0}
        state_dict = {'eMBB': [], 'mMTC': [], 'URLLC': []}

        for slice in self.slices:
            slice.slice_capacity = (1 + self.actions[action])*slice.slice_capacity

        for user_request in users_requests:
            for slice in self.slices:
                if slice.accept_user(user_request):
                    reward_dict[slice.name] = +1
                elif slice.accept_user(user_request) is None:
                    reward_dict[slice.name] = -5
                else:
                    reward_dict[slice.name] = -1

        state = []
        for slice in self.slices:
            connected_ratio, req_b2_ratio, bw_remaining = slice.setState(self.total_capcity, n_users)
            state_dict[slice.name].append((connected_ratio, req_b2_ratio, bw_remaining))
            state.append(connected_ratio)
            state.append(req_b2_ratio)
            state.append(bw_remaining)

        state = np.array(state)

        reward = sum([reward_dict[slice] for slice in reward_dict]) if self.isTerminalState(state) else -10
        return state, action, reward, (state_dict, reward_dict)

    def reset(self, slice_info):
        self.distribute_bandwidth(slice_info)

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)


if __name__ == "__main__":
    """
    BASESTATION_CAP: 10 GB
    RATIOS:
        eMBB -> 0.45
        mMTC -> 0.25
        URLLC -> 0.30
    USER LIMITS:
        LOW LIMIT -> 40 Mbps
        UPPER LIMIT -> 1 Gbps
    """
    TOTAL_CAPACITY = 10000000000
    SLICE_INFO = {'eMBB': [0.45, (700000000, 1000000000)],
                   'mMTC': [0.25, (150000000, 600000000)],
                    'URLLC': [0.3, (40000000, 100000000)]}
    LOWER_LIM = 40000000
    UPPER_LIM = 1000000000

    env = Network(total_capcity=TOTAL_CAPACITY, slice_info=SLICE_INFO,
                  action_factor=1.0, low_limit=LOWER_LIM, high_limit=UPPER_LIM)

    for _ in range(10):
        action = env.actionSpaceSample()
        print(env.step(action))









