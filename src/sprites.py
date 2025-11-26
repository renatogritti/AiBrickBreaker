import pygame
from src.settings import *

class Paddle(pygame.sprite.Sprite):
    """Representa a raquete controlada pelo jogador no jogo."""

    def __init__(self):
        """Inicializa a raquete, definindo sua aparência, posição inicial e velocidade."""
        super().__init__()
        self.image = pygame.Surface([PADDLE_WIDTH, PADDLE_HEIGHT])
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - 10
        self.speed = PADDLE_SPEED

    def update(self):
        """Atualiza a posição da raquete com base na entrada do teclado e a mantém dentro dos limites da tela."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed

        # Mantém a raquete dentro da tela
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

class Ball(pygame.sprite.Sprite):
    """Representa a bola no jogo."""

    def __init__(self):
        """Inicializa a bola, definindo sua aparência, posição inicial e velocidade."""
        super().__init__()
        self.image = pygame.Surface([BALL_RADIUS * 2, BALL_RADIUS * 2])
        self.image.set_colorkey(BLACK) # Torna o fundo transparente
        pygame.draw.circle(self.image, WHITE, (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.centery = SCREEN_HEIGHT // 2
        self.speed_x = BALL_SPEED_X
        self.speed_y = BALL_SPEED_Y

    def update(self):
        """Atualiza a posição da bola com base em suas velocidades X e Y."""
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

class Brick(pygame.sprite.Sprite):
    """Representa um tijolo no jogo."""

    def __init__(self, x, y, color, is_special=False):
        """Inicializa um tijolo, definindo sua posição, cor e se é um tijolo especial.

        Args:
            x (int): A coordenada X superior esquerda do tijolo.
            y (int): A coordenada Y superior esquerda do tijolo.
            color (tuple): A cor RGB do tijolo.
            is_special (bool, optional): True se o tijolo for especial (e.g., libera power-up). Padrão para False.
        """
        super().__init__()
        self.is_special = is_special
        self.image = pygame.Surface([60, 20])
        if self.is_special:
            self.image.fill(YELLOW) # Cor para tijolo especial
        else:
            self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
