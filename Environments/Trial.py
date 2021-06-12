import pygame

pygame.init()

screen = pygame.display.set_mode((200, 200))
run = True
pos = pygame.Vector2(100, 100)
clock = pygame.time.Clock()

# speed of your player
speed = 2

# key bindings
move_map = {pygame.K_LEFT: pygame.Vector2(-1, 0),
            pygame.K_RIGHT: pygame.Vector2(1, 0),
            pygame.K_UP: pygame.Vector2(0, -1),
            pygame.K_DOWN: pygame.Vector2(0, 1)}

while run:
  for e in pygame.event.get():
    if e.type == pygame.QUIT: run = False

  screen.fill((30, 30, 30))
  # draw player, but convert position to integers first
  pygame.draw.circle(screen, pygame.Color('dodgerblue'), [int(x) for x in pos], 10)
  pygame.display.flip()

  # determine movement vector
  pressed = pygame.key.get_pressed()
  move_vector = pygame.Vector2(0, 0)
  for m in (move_map[key] for key in move_map if pressed[key]):
    move_vector += m

  # normalize movement vector if necessary
  if move_vector.length() > 0:
    move_vector.normalize_ip()

  # apply speed to movement vector
  move_vector *= speed



  # update position of player
  pos += move_vector

  clock.tick(60)