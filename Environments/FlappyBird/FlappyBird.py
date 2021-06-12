"""
@Author: Ishaan Roy (adapted from TechwithTim)
File contains: Flappy bird environment
TODO:
------------------------------------------
=> Complete all game graphics and initialise all classes in the main loop

------------------------------------------
ChangeLog
------------------------------------------
=> Version 1.0.0
    -> Added classes for bird, pipe and base
        -> Bird class: movement mechanics (jump, tilt)
        -> Pipe class: movement mechanics (collision detection)
        -> Base class: movement mechanics (1-d motion mechanics)
    -> Added main game loop
=> Version 1.0.1
    -> Modified draw window function:
        -> Included pipes and base
=> Version 1.1.1
    -> Modified the main loop to include random generation of pipes
    -> Basic game mechanics done
=> Version 1.1.2
    -> Added score on screen
=> Version 2.1.2
    -> Added AI agent to controll jump actions
    -> Run method written
    -> Fittness function defined

------------------------------------------
NEAT Information
------------------------------------------
-> Inputs:
    (Bird.y, distance_from_topPipe, distance_from_bottomPipe)
-> Outputs:
    (Jump, Not Jump)
-> Activation: tanh
-> Population size: 100
-> Fitness Function: Max distance covered by bird
-> Max generations: 30
------------------------------------------
"""
import pygame
import neat
import time
import os
import random
import numpy as np

pygame.font.init()
WIN_WIDTH = 500
WIN_HEIGHT = 800

# Loading Images
IMG_DIR = './imgs'
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join(IMG_DIR, 'bird1.png'))),
             pygame.transform.scale2x(pygame.image.load(os.path.join(IMG_DIR, 'bird2.png'))),
             pygame.transform.scale2x(pygame.image.load(os.path.join(IMG_DIR, 'bird3.png')))]

PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(IMG_DIR, 'pipe.png')))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(IMG_DIR, 'bg.png')))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join(IMG_DIR, 'base.png')))
STAT_FONT = pygame.font.SysFont('comicsans', 50)

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VELOCITY = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def Jump(self):
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y

    def Move(self):
        self.tick_count += 1
        distance = self.velocity * self.tick_count + 1.5 * self.tick_count ** 2
        if distance >= 16:
            distance = 16
        if distance < 0:
            distance -= 2

        self.y = self.y + distance
        if distance < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VELOCITY

    def Draw(self, window):
        self.img_count += 1
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_img = pygame.transform.rotate(self.img, self.tilt)
        new_rectangle = rotated_img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        window.blit(rotated_img, new_rectangle.topleft)

    def GetMask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200
    VELOICTY = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.SetHeight()

    def SetHeight(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def Move(self):
        self.x -= self.VELOICTY

    def Draw(self, window):
        window.blit(self.PIPE_TOP, (self.x, self.top))
        window.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def Collide(self, bird):
        bird_mask = bird.GetMask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return False


class Base:
    VELOCITY = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):

        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def Move(self):
        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def Draw(self, window):
        window.blit(self.IMG, (self.x1, self.y))
        window.blit(self.IMG, (self.x2, self.y))



def draw_window(window, birds, pipes, base, score):
    window.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.Draw(window)

    text = STAT_FONT.render('Score: ' + str(score), 1, (255, 255, 255))
    window.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    base.Draw(window)
    for bird in birds:
        bird.Draw(window)
    pygame.display.update()


def main(genomes, config):
    pygame.init()
    nets = []
    genes = []
    birds = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        genome.fitness = 0
        genes.append(genome)

    base = Base(730)
    pipes = [Pipe(600)]
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0

    running = True
    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x + pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            running = False
            break

        for x, bird in enumerate(birds):
            bird.Move()
            genes[x].fitness += 0.1

            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height),
                                                       abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] > 0.5:
                bird.Jump()


        add_pipe = False
        removed = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.Collide(bird):
                    genes[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    genes.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                removed.append(pipe)
            pipe.Move()

        if add_pipe:
            score += 1
            for gen in genes:
                gen.fitness += 5
            pipes.append(Pipe(600))

        for removed_pipe in removed:
            pipes.remove(removed_pipe)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                genes.pop(x)

        base.Move()
        # bird.Move()
        draw_window(window, birds, pipes, base, score)

def Run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    population = neat.Population(config)
    stats = neat.StatisticsReporter()
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(stats)

    winner = population.run(main, 50)



if __name__ == "__main__":
    local_directory = os.path.dirname(__file__)
    config_path = os.path.join(local_directory, 'Configuration')

    Run(config_path)
