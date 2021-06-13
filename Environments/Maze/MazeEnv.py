"""
@Author: Ishaan Roy adpated from https://github.com/MattChanTK/
File contains: Simple Maze Environment
TODO:
-----------------------------------------
=> Add class for rendering the environment
-----------------------------------------
ChangeLog
-----------------------------------------
=> Version 0.0
    -> Simple Maze Environment
    -> Loops: Multiple paths supported
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
    DIR_PATH = os.path.dirname(__file__)

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

        # if self.num_portals > 0:
        #     self.__set_random_portals(self.num_portals, 2)

    def __break_random_walls(self, percent):
        num_cells_to_break = int(round(self.getMazeH * self.getMazeW * percent))
        cell_ids = random.sample(range(self.getMazeW * self.getMazeW), num_cells_to_break)

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

    @staticmethod
    def all_walls_intact(cell):
        return cell & 0xF == 0

    def save_maze(self, file_name):
        dir_path = os.path.join(self.DIR_PATH, 'Examples')
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            pass
        file_name = os.path.join(dir_path, file_name)
        np.save(file_name, self.maze_cells, allow_pickle=False, fix_imports=True)

    @classmethod
    def load_maze(cls, file_name):
        parent_path = os.path.join(cls.DIR_PATH, 'Examples')
        file_path = os.path.join(parent_path, file_name)
        if not os.path.exists(file_path):
            raise ValueError('File Does Not Exist')
        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)


class MazeRenderer:
    WHITE = (0, 0, 0)
    BLACK = (255, 255, 255)
    RED = (255, 0, 0)

    def __init__(self, maze_name, maze_file_path, maze_size,
                 screen_size, has_loops=True, num_portals=0,
                 enable_render=True):
        # Pygame Config
        pygame.init()
        pygame.display.set_caption(maze_name)
        self.clock = pygame.time.Clock()
        self.__game_over = False
        self.__enable_render = enable_render

        if maze_file_path is None:
            self.__maze = Maze(maze_size=maze_size, has_loops=has_loops, num_portals=num_portals)
        else:
            if not os.path.exists(maze_file_path):
                raise FileNotFoundError
            else:
                self.__maze = Maze(maze_cells=Maze.load_maze(maze_file_path))

        self.maze_size = self.__maze.maze_size
        if self.__enable_render:
            self.screen = pygame.display.set_mode(screen_size)
            self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        # Maze Start point
        self.__start_pt = np.zeros(2, dtype=int)
        # Maze End Point
        self.__end_pt = np.array(self.__screen_size)
        # Initialise Agent Position
        self.__robot = self.entrance

        if self.__enable_render:
            self.background = pygame.Surface(self.screen.get_size()).convert()
            self.background.fill(self.BLACK)

            self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
            self.maze_layer.fill((0, 0, 0, 0,))

            # Show the maze
            self.__draw_maze()

            # Show the agent
            self.__draw_robot()

            # Show the entrance
            self.__draw_entrance()

            # Show the end point
            self.__draw_endpoint()

    def __draw_maze(self):
        pass

    def __draw_robot(self):
        pass

    def __draw_entrance(self):
        pass

    def __draw_endpoint(self):
        pass

    @property
    def maze(self):
        return self.__maze

    @property
    def entrance(self):
        return self.__start_pt

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_WIDTH(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_HEIGHT(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_WIDTH(self):
        return float(self.SCREEN_WIDTH / self.maze.getMazeW)

    @property
    def CELL_HEIGHT(self):
        return float(self.SCREEN_HEIGHT / self.maze.getMazeH)
