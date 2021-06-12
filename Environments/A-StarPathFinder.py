import pygame
import numpy
from heapq import *
import sys
from pygame.locals import *
import random

class Grid:
    def __init__(self, game):
        self.game = game
        self.length = self.game.screen_res[0] / 15
        self.width = (self.game.screen_res[1] / 15) - 3

        self.nodes = [[Node(self, [row, col + 3]) for row in range(self.length)] for col in range(self.width)]

    def update(self):
        for col in self.nodes:
            for node in col:
                node.update()
                node.draw(self.game.screen)

        for i in range(self.length):
            pygame.draw.line(self.game.screen, [100] * 3, (15 * i, 45), (15 * i, 495))

        for i in range(self.width):
            pygame.draw.line(self.game.screen, [100] * 3, (0, (15 * i) + 45), (750, (15 * i) + 45))

    def clearPath(self):
        for col in self.nodes:
            for node in col:
                if node.in_path:
                    node.in_path = False
                    node.color = 0


class Node():
    def __init__(self, grid, pos):
        self.grid = grid
        self.game = self.grid.game

        self.pos = pos
        self.blit_pos = [i * 15 for i in self.pos]
        self.color = [0, 0, 0]

        self.image = pygame.Surface((15, 15))

        self.rect = self.image.get_rect(topleft=self.blit_pos)

        self.solid = 0
        self.in_path = False
        self.checked = False

    def update(self):
        if self.checked and self.game.show_checked:
            self.color = [0, 255, 0]
        if self.checked and self.game.show_checked == False:
            self.color = [0, 0, 0]

        if self.in_path:
            self.color = [0, 0, 255]

        if self.game.pathing:
            pass

        else:
            if pygame.mouse.get_pressed()[0] and self.rect.collidepoint(self.game.mpos):
                self.solid = 1
                self.in_path = False
                self.color = [255, 0, 0]

            if pygame.mouse.get_pressed()[2] and self.rect.collidepoint(self.game.mpos):
                self.solid = 0
                self.in_path = False
                self.color = [0, 0, 0]

    def draw(self, screen):
        self.image.fill(self.color)
        screen.blit(self.image, self.rect)



def heuristic(a, b, pathing):
    x = abs(a[0] - b[0])
    y = abs(a[1] - b[1])

    if pathing == '*':

        if x > y:
            return 14 * y + 10 * (x - y)
        else:
            return 14 * x + 10 * (y - x)
    else:
        return 10 * (x + y)


def astar(array, start, goal, pathing):
    if pathing == '+':
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    else:
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal, pathing)}
    oheap = []
    checked = []

    heappush(oheap, (fscore[start], start))

    while oheap:

        current = heappop(oheap)[1]
        checked.append(current)

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]

            return list(reversed(data)), checked

        array[current[0], current[1]] = 2
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor, pathing)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array.flat[array.shape[1] * neighbor[0] + neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if array[neighbor[0]][neighbor[1]] == 2 and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal, pathing)
                heappush(oheap, (fscore[neighbor], neighbor))

    return False


pygame.init()


class Game():
    def __init__(self):
        # window setup
        pygame.display.set_caption('A* Visual')

        # initiate the clock and screen
        self.clock = pygame.time.Clock()
        self.last_tick = pygame.time.get_ticks()
        self.screen_res = [750, 495]

        self.font = pygame.font.SysFont("Calibri", 16)

        self.screen = pygame.display.set_mode(self.screen_res, pygame.HWSURFACE, 32)
        self.pathing = False
        self.pathing_type = '*'
        self.show_checked = False

        self.grid = Grid(self)

        while 1:
            self.Loop()

    def Run(self):
        self.pathing = True
        self.grid.clearPath()

        node_array = self.Convert()

        path, check = astar(node_array, (0, 0), (29, 49), self.pathing_type)

        for pos in check:
            self.grid.nodes[pos[0]][pos[1]].checked = True

        if path != False:
            for pos in path:
                self.grid.nodes[pos[0]][pos[1]].in_path = True
        else:
            pass

        print(len(path))
        self.pathing = False

    def Clear(self):
        self.grid = Grid(self)

    def Convert(self):
        array = [[self.grid.nodes[col][row].solid for row in range(self.grid.length)] for col in
                 range(self.grid.width)]
        nodes = numpy.array(array)
        return nodes

    def blitInfo(self):
        text = self.font.render("Press Enter to find path, press Space to clear board", 1, (255, 255, 255))
        text2 = self.font.render("Press c to toggle checked nodes, and 1 and 2 to switch pathing types", 1,
                                 (255, 255, 255))

        check = self.font.render("Checked nodes: " + str(self.show_checked), 1, (255, 255, 255))
        ptype = self.font.render("Pathing type: " + self.pathing_type, 1, (255, 255, 255))

        self.screen.blit(text, (5, 5))
        self.screen.blit(text2, (5, 25))

        self.screen.blit(check, (500, 5))
        self.screen.blit(ptype, (500, 25))

    def Loop(self):
        # main game loop
        self.eventLoop()

        self.Tick()
        self.Draw()
        pygame.display.update()

    def eventLoop(self):
        # the main event loop, detects keypresses
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.Run()
                if event.key == K_SPACE:
                    self.Clear()
                if event.key == K_1:
                    self.pathing_type = '+'
                if event.key == K_2:
                    self.pathing_type = '*'
                if event.key == K_c:
                    print
                    self.show_checked
                    if self.show_checked:
                        self.show_checked = False
                    else:
                        self.show_checked = True

    def Tick(self):
        # updates to player location and animation frame
        self.ttime = self.clock.tick()
        self.mpos = pygame.mouse.get_pos()

    def Draw(self):
        self.screen.fill(0)
        self.grid.update()
        self.blitInfo()

if __name__ == "__main__":
    Game()