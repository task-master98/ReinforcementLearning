"""
@Author: Ishaan Roy adpated from https://github.com/MattChanTK/
File contains: Simple Maze Environment
TODO:
-----------------------------------------

-----------------------------------------
ChangeLog
-----------------------------------------

-----------------------------------------
"""
import numpy as np
import random
import pygame
import os


class Maze:
    DIRECTIONS = {
        'N': (0, -1),
        'S': (0, 1),
        'E': (1, 0),
        'W': (-1, 0)
    }

    def __init__(self, maze_cells=None, maze_size=(10, 10), has_loops=True, num_portals=0):
        # Maze variables
        self.maze_cells = maze_cells
        self.maze_size = maze_size
        self.has_loops = has_loops
        self.__portals_dict = dict()
        self.__portals = []
        self.num_portals = num_portals

        # Generate Maze
        self._generate_maze()

    def _generate_maze(self):
        self.maze_cells = np.zeros(self.maze_size, dtype=int)
        current_cell = (random.randint(0, self.getMazeW),
                        random.randint(0, self.getMazeH))
        num_cells_visited = 1
        cell_stack = [current_cell]

        while cell_stack:
            current_cell = cell_stack.pop()
            x0, y0 = current_cell
            neighbors = {}
            for dir_key, dir_vec in self.DIRECTIONS.items():
                x1 = x0 + dir_vec[0]
                y1 = y0 + dir_vec[1]
                if self.is_within_bounds(x1, y1):
                    if self.all_walls_intact(self.maze_cells[x1, y1]):
                        neighbors[dir_key] = (x1, y1)

            if neighbors:
                random_dir = random.choice(tuple(neighbors.keys()))
                x1, y1 = neighbors[random_dir]

                # Break walls between neighbor and current cell
                self.maze_cells[x1, y1] = self.__break_walls(self.maze_cells[x1, y1],
                                                             self.__get_opposite_wall(random_dir))
                # current cell pushed into the stack
                cell_stack.append(current_cell)

                # neighbor cell to be the current cell
                cell_stack.append((x1, y1))

                num_cells_visited += 1

        if self.has_loops:
            self.__break_random_walls(0.2)

        if self.num_portals > 0:
            self.__set_random_portals(self.num_portals, 2)

    def __break_random_walls(self, percent):
        num_cells_to_break = int(round(self.getMazeH*self.getMazeW*percent))
        cell_ids = random.sample(range(self.getMazeW*self.getMazeW), num_cells_to_break)

        for cell_id in cell_ids:
            x = cell_id % self.getMazeW
            y = int(cell_id / self.getMazeH)

            random_dir_order = random.sample(self.DIRECTIONS.keys(), 4)
            for dir in random_dir_order:
                # Break wall if not already open
                if self.is_breakable((x, y), dir):
                    self.maze_cells[x, y] = self.__break_walls(self.maze_cells[x, y], dir)
                    break

    def is_breakable(self, cell_id, dir):
        x1 = cell_id[0] + self.DIRECTIONS[dir][0]
        y1 = cell_id[1] + self.DIRECTIONS[dir][1]

        return not self.is_open(cell_id, dir) and self.is_within_bounds(x1, y1)

    def is_open(self, cell_id, dir):
        x1 = cell_id[0] + self.DIRECTIONS[dir][0]
        y1 = cell_id[1] + self.DIRECTIONS[dir][1]

        if self.is_within_bounds(x1, y1):
            this_wall = bool(self.get_walls_status(self.maze_cells[cell_id[0], cell_id[1]])[dir])
            other_wall = bool(self.get_walls_status(self.maze_cells[x1, y1])[self.__get_opposite_wall(dir)])
            return this_wall or other_wall
        return False

    def get_walls_status(self, cell):
        walls = {
            'N': (cell & 0x1) >> 0,
            'E': (cell & 0x2) >> 1,
            'S': (cell & 0x4) >> 2,
            'W': (cell & 0x8) >> 3
        }
        return walls

    def is_within_bounds(self, x, y):
        return 0 <= x < self.getMazeW and 0 <= y < self.getMazeH

    @staticmethod
    def __break_walls(cell, direction):
        if "N" in direction:
            cell |= 0x1
        if "E" in direction:
            cell |= 0x2
        if "S" in direction:
            cell |= 0x4
        if "W" in direction:
            cell |= 0x8
        return cell

    @staticmethod
    def __get_opposite_wall(dirs):
        opposite_dirs = ""
        for dir in dirs:
            if dir == "N":
                opposite_dir = "S"
            elif dir == "S":
                opposite_dir = "N"
            elif dir == "E":
                opposite_dir = "W"
            elif dir == "W":
                opposite_dir = "E"
            else:
                raise ValueError("Not a Valid Direction")

            opposite_dirs += opposite_dir
        return opposite_dirs

    @property
    def getMazeW(self):
        return int(self.maze_size[0])

    @property
    def getMazeH(self):
        return int(self.maze_size[1])




