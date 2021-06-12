"""
@Author: Ishaan Roy
File contains: Network Environment
TODO:
------------------------------------------
=> Create network simulation in pygame:
    -> Add relevant objects: Base station, users, slices
    -> Add relevant methods
=> Ambiguity in connected status? Fix
=> Work out how to disconnect UEs from the base station:
    -> Where should the method be included?
    -> Which parameters should change?
------------------------------------------
ChangeLog
------------------------------------------

------------------------------------------
"""
import pygame
import random
import numpy as np
import os
from collections import deque
from enum import Enum
import sys
from array import array

IMG_DIR = './Imgs'
BASESTATION_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(IMG_DIR, 'signal-tower.png')))
MOBILE_IMG = pygame.image.load(os.path.join(IMG_DIR, 'cell-phone.png'))
VEHICLE_IMG = pygame.image.load(os.path.join(IMG_DIR, 'car.png'))
IOT_DEVICE_IMG = pygame.image.load(os.path.join(IMG_DIR, 'cpu.png'))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(IMG_DIR, 'bg.png')))

# Colors
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)

# Display Settings
pygame.font.init()
WIN_WIDTH = 800
WIN_HEIHT = 800
font = pygame.font.SysFont('comicsans', 25)

# Simulation Parameters
MB = 1000000
GB = 1000000000

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class NetworkSlice:
    def __init__(self, name, slice_ratio,  BS_capacity, max_capacity=100):
        self.name = name
        self.slice_ratio = slice_ratio
        self.queue = deque(maxlen=max_capacity)
        self.initialise_slice_capacity(BS_capacity)
        self.connected_users = 0
        self.requested_bw = 0

    def initialise_slice_capacity(self, BS_bw_capacity):
        self.capacity = self.slice_ratio*BS_bw_capacity

    def accept_user(self, user, requested_bw):
        if len(self.queue) != 0:
            for user in self.queue:
                user.usage_time += 1
        self.queue.append(user)
        self.capacity -= requested_bw
        self.connected_users += 1
        # if self.connected_users > self.queue.maxlen:
        #     self.connected_users = 0
        self.requested_bw += requested_bw


    def remove_user(self, client):
        self.queue.remove(client)
        self.capacity += client.bandwidth
        self.connected_users -= 1
        return client

    def queue_status(self):
        queue_status = len(self.queue) == self.queue.maxlen
        if queue_status == True:
            return False
        return True

    def is_resource_available(self, requested_bw):
        return requested_bw >= self.capacity



class BaseStation:
    IMG = BASESTATION_IMG
    def __init__(self, x, y, coverage, capacity, slice_info):
        self.x = x
        self.y = y
        self.coverage = coverage
        self.capacity = capacity
        self.initialise_slices(slice_info)
        self.img = self.IMG

    def initialise_slices(self, slice_info):
        self.slices = []
        for key in slice_info:
            slice_instance = NetworkSlice(key, slice_info[key], self.capacity)
            self.slices.append(slice_instance)

    def disconnect_UEs(self):
        """
        TODO: This function is to be modified to receive an action
              from the RL agent
        :return:
        """
        for slice in self.slices:
            for user in list(slice.queue):
                if user.usage_time > 0:
                    slice.queue.popleft()
                    user.connected = False


    def draw(self, window):
        window.blit(self.img, (self.x, self.y))


class Client:
    VELOCITY = 10
    VEHICLE = VEHICLE_IMG
    IoT = IOT_DEVICE_IMG
    MOBILE = MOBILE_IMG
    def __init__(self, id, x, y, UE_type):
        """

        :param x: UE coordinate x
        :param y: UE coordinate y
        :param UE_type: Type os UE
                        Type 0 -> Mobile
                        Type 1 -> Vehicle
                        Type 2 -> IoT device
        """
        self.id = id
        self.x = x
        self.y = y
        self.allocate_slice(UE_type)
        self.bandwidth = 0
        self.connected_slice = None
        self.connected = False
        self.blocked = False
        self.usage_time = 0


    def allocate_slice(self, UE_type):
        self.UE_type = UE_type
        if self.UE_type == 0:
            self.img = self.MOBILE
            self.slice = 'eMBB'
        elif self.UE_type == 1:
            self.img = self.VEHICLE
            self.slice = 'URLLC'
        else:
            self.img = self.IoT
            self.slice = 'mMTC'



    def move(self):
        if self.x >= WIN_WIDTH - 32 or self.x < 0:
            self.x = random.randint(WIN_WIDTH//2, WIN_WIDTH)
        if self.y >= WIN_HEIHT - 32 or self.y < 0:
            self.y = random.randint(WIN_HEIHT//2, WIN_HEIHT)

        clockwise_dir = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        random_dir = random.choice(clockwise_dir)
        if random_dir == Direction.RIGHT:
            self.x += self.VELOCITY
        elif random_dir == Direction.LEFT:
            self.x -= self.VELOCITY
        elif random_dir == Direction.UP:
            self.y -= self.VELOCITY
        else:
            self.y += self.VELOCITY

    def is_in_coverage(self, basestation: BaseStation):
        UE_location = np.array([self.x, self.y])
        BS_location = np.array([basestation.x, basestation.y])
        BS_to_UE = np.linalg.norm(UE_location - BS_location)
        if np.sqrt(BS_to_UE) <= basestation.coverage:
            return True
        return False

    def request_connection(self):

        if self.UE_type == 0:
            requested_bandwidth = np.random.normal(loc=1*GB, scale=200*MB)
        elif self.UE_type == 1:
            requested_bandwidth = np.random.normal(loc=600*MB, scale=200*MB)
        else:
            requested_bandwidth = np.random.normal(loc=30*MB, scale=15*MB)
        return requested_bandwidth

    def connect_to_BS(self, basestation: BaseStation):
        coverage_status = self.is_in_coverage(basestation)
        if coverage_status:
            requested_bw = self.request_connection()
            self.bandwidth = requested_bw
            for slice in basestation.slices:
                if self.slice == slice.name:
                    if slice.queue_status() and slice.is_resource_available(requested_bw):
                        slice.accept_user(self, requested_bw)
                        self.connected = True
                        self.connected_slice = slice
                        # self.usage_time += 1
                    else:
                        self.blocked = True

        else:
            self.connected = False

    def disconnect_from_BS(self, basestation):
        if not self.is_in_coverage(basestation) and self.connected:
            self.connected_slice.remove_user(self)
            self.connected = False
        elif self.connected and self.usage_time > 0:
            self.connected = False


    def draw(self, window, basestation):
        window.blit(self.img, (self.x, self.y))
        if self.connected:
            pygame.draw.line(window, RED, (self.x+16,self.y+16),
                             (basestation.x+16, basestation.y+16))
        else:
            pygame.draw.line(window, BLACK, (self.x + 16, self.y + 16),
                             (basestation.x + 16, basestation.y + 16))




def draw_window(window, users, basestation):
    window.fill(WHITE)
    basestation.draw(window)
    pygame.draw.circle(window, color=BLUE, center=(basestation.x+16, basestation.y+16),
                       radius=basestation.coverage, width=2)
    for user in users:
        user.draw(window, basestation)

    eMBB_connected_users = basestation.slices[0].connected_users
    eMBB_requested_bandwidth = basestation.slices[0].requested_bw
    eMBB_remaining_bandwidth = basestation.slices[0].capacity

    URLLC_connected_users = basestation.slices[1].connected_users
    URLLC_requested_bandwidth = basestation.slices[1].requested_bw
    URLLC_remaining_bandwidth = basestation.slices[1].capacity

    mMTC_connected_users = basestation.slices[2].connected_users
    mMTC_requested_bandwidth = basestation.slices[2].requested_bw
    mMTC_remaining_bandwidth = basestation.slices[2].capacity

    eMBB_text = font.render('eMBB Connected Users: ' + str(eMBB_connected_users), True, BLACK)
    URLLC_text = font.render('URLLC Connected Users: ' + str(URLLC_connected_users), True, BLACK)
    mMTC_text = font.render('mMTC Connected Users: ' + str(mMTC_connected_users), True, BLACK)
    window.blit(eMBB_text, (0, 0))
    window.blit(URLLC_text, (0, 25))
    window.blit(mMTC_text, (0, 50))
    pygame.display.update()

def main():
    pygame.init()

    NUM_CLIENTS = 10
    users = []
    UE_type = [0, 1, 2]
    SLICE_INFO = {'eMBB': 0.70, 'URLLC': 0.25, 'mMTC': 0.05}


    for idx in range(NUM_CLIENTS):
        client = Client(id=idx, x=random.randint(0, WIN_WIDTH), y=random.randint(0, WIN_HEIHT),
                        UE_type=random.choice(UE_type))
        users.append(client)
    base_station = BaseStation(x=WIN_WIDTH//2, y=WIN_HEIHT//2, capacity=0, coverage=200, slice_info=SLICE_INFO)

    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIHT))
    clock = pygame.time.Clock()

    running = True

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event == pygame.QUIT:
                # running = False
                pygame.quit()
                sys.exit()

        for client in users:
            client.move()
            client.connect_to_BS(base_station)
            # client.disconnect_from_BS(base_station)

        base_station.disconnect_UEs()

        draw_window(window, users, base_station)

if __name__ == "__main__":
    main()