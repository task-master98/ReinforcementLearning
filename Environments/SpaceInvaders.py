"""
@Author: Ishaan Roy
File contains: Space Invader game
TODO:
------------------------------------------
=> Add enemy to the game
=> Add firing mechanism to the game
=> Refactor player and enemy as classes instead
=> Set play again button
=> Limit amount of bullets
=> Multiple pleyers
------------------------------------------
ChangeLog
------------------------------------------
=> Version 1.0.0
    -> Added player to the game
    -> Added movement mechanism to the player
=> Version 1.1.0
    -> Enemy has been added
    -> Randomised initial coordinates of enemy
    -> Movement mechanics of enemy added
=> Version 1.2.0
    -> Collision mechanics have been added
    -> Respawing of enemy at random locations enabled
    -> Bug fix: Incorrect respawning fixed
=> Version 1.2.1
    -> Score board added to the game
=> Version 1.3.1
    -> Multiple enemies added
    -> Respawning mechanics of multiple enemies added
=> Version 1.3.2
    -> Game over condition added
------------------------------------------

"""
import pygame
import random
import math
import numpy as np

## Initialise game instance
pygame.init()
SCREEN = pygame.display.set_mode((800, 600))

# Title and icon
pygame.display.set_caption('Space Invaders')
icon = pygame.image.load('ufo.png')
pygame.display.set_icon(icon)

# Score
SCORE = 0
font = pygame.font.Font('freesansbold.ttf', 32)
text_X = 10
text_Y = 10
over_font = pygame.font.Font('freesansbold.ttf', 64)

def ShowScore(x, y):
    score = font.render('Score: ' + str(SCORE), True, (255, 255, 255))
    SCREEN.blit(score, (x, y))

# Game Over Text
def GameOver(x, y):
    go_text = over_font.render('GAME OVER', True, (255, 255, 255))
    SCREEN.blit(go_text, (x, y))
# Player
player_img = pygame.image.load('space-invaders.png')
player_X = 370
player_Y = 480
delta_X = 0.3
X_change = 0
movement_map = {
    pygame.K_LEFT: pygame.Vector2(-1, 0),
    pygame.K_RIGHT: pygame.Vector2(1, 0),
    pygame.K_DOWN: pygame.Vector2(0, 1),
    pygame.K_UP: pygame.Vector2(0, -1)
}

def MovePlayer(x, y):
    SCREEN.blit(player_img, (x, y))

# Enemy
NUM_ENEMIES = 6
enemy_img = []
enemy_X = []
enemy_Y = []
enemy_Xchange = []
enemy_Ychange = []

for _ in range(NUM_ENEMIES):
    enemy_img.append(pygame.image.load('alien.png'))
    enemy_X.append(random.randint(0, 800-32))
    enemy_Y.append(random.randint(50, 150))
    enemy_Xchange.append(0.3)
    enemy_Ychange.append(10)

def MoveEnemy(x, y, i):
    SCREEN.blit(enemy_img[i], (x, y))

# Bullet
bullet_img = pygame.image.load('bullet.png')
bullet_X = 0
bullet_Y = 480
bullet_Xchange = 0
bullet_Ychange = 1
bullet_state = 'ready'

def FireBullet(x, y):
    global bullet_state
    bullet_state = 'fire'
    SCREEN.blit(bullet_img, (x+16, y+10))

def IsCollision(enemy_X, enemy_Y, bullet_X, bullet_Y):
    enemy_vector = np.array([enemy_X, enemy_Y])
    bullet_vector = np.array([bullet_X, bullet_Y])
    norm =  math.sqrt(np.linalg.norm(enemy_vector - bullet_vector))
    # print(norm)
    if norm < 6:
        return True
    return False

# Game Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                # print('Left Key is Pressed')
                X_change = -0.3
            if event.key == pygame.K_RIGHT:
                # print('Right Key is pressed')
                X_change = 0.3
            if event.key == pygame.K_SPACE:
                if bullet_state == 'ready':
                    bullet_X = player_X
                    FireBullet(bullet_X, bullet_Y)
        if event.type == pygame.KEYUP:
            X_change = 0


    SCREEN.fill((0, 0, 0))
    player_X += X_change


    if player_X < 0:
        player_X = 0
    elif player_X >= 800 - 32:
        player_X = 800 - 32

    # Enemy Movement
    for i in range(NUM_ENEMIES):
        if enemy_Y[i] > 200:
            for j in range(NUM_ENEMIES):
                enemy_Y[j] = 2000
            GameOver(200, 250)
            break


        enemy_X[i] += enemy_Xchange[i]

        if enemy_X[i] < 0:
            enemy_Xchange[i] = 0.3
            enemy_Y[i] += enemy_Ychange[i]
        elif enemy_X[i] >= 800 - 32:
            enemy_Xchange[i] = -0.3
            enemy_Y[i] += enemy_Ychange[i]
        # Collision
        collision = IsCollision(enemy_X[i], enemy_Y[i], bullet_X, bullet_Y)

        if collision:
            bullet_Y = 480
            bullet_state = 'ready'
            SCORE += 1
            # print(SCORE)
            enemy_X[i] = random.randint(0, 800 - 32)
            enemy_Y[i] = random.randint(50, 150)

        MoveEnemy(enemy_X[i], enemy_Y[i], i)

    # Bullet Movement
    if bullet_Y <= 0:
        bullet_X = player_X + 16
        bullet_Y = player_Y + 10
        bullet_state = 'ready'
    if bullet_state == 'fire':
        FireBullet(bullet_X, bullet_Y)
        bullet_Y -= bullet_Ychange

    MovePlayer(player_X, player_Y)
    ShowScore(text_X, text_Y)
    pygame.display.flip()




