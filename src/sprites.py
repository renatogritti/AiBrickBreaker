"""
-----------------------------------------------------------------------
Arquivo: src/sprites.py
Data: 27/11/2025
Versão: 1.0
Autor: Renato Gritti
Descrição:
    Define as classes de Sprites do jogo (Raquete, Bola, Tijolo) herdando
    de pygame.sprite.Sprite. Contém a lógica de movimento e renderização
    básica de cada entidade.
-----------------------------------------------------------------------
"""

import pygame
from src.config import *

class Paddle(pygame.sprite.Sprite):
    """
    Representa a raquete controlada pelo jogador ou agente.
    """

    def __init__(self):
        """
        Inicializa a raquete, definindo sua aparência, posição inicial e velocidade.
        """
        super().__init__()
        self.image = pygame.Surface([PADDLE_WIDTH, PADDLE_HEIGHT])
        self.image.fill(PADDLE_COLOR)
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - PADDLE_START_Y_OFFSET
        self.speed = PADDLE_SPEED

    def update(self, action=None):
        """
        Atualiza a posição da raquete.
        
        Args:
            action (int, optional): Ação do agente (0=Ficar, 1=Esquerda, 2=Direita).
                                    Se None, usa entrada do teclado (Humano).
        """
        dx = 0
        if action is None:
            # Controle por Teclado (Humano)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                dx = -self.speed
            if keys[pygame.K_RIGHT]:
                dx = self.speed
        else:
            # Controle por IA
            if action == 1: # Esquerda
                dx = -self.speed
            elif action == 2: # Direita
                dx = self.speed
            # action == 0 faz nada (Ficar parado)

        self.rect.x += dx
        
        # Armazena velocidade atual para física da bola
        self.current_vel_x = dx


        # Mantém a raquete dentro dos limites da tela
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

class Ball(pygame.sprite.Sprite):
    """
    Representa a bola no jogo.
    """

    def __init__(self):
        """
        Inicializa a bola, definindo sua aparência e posição centralizada.
        """
        super().__init__()
        self.image = pygame.Surface([BALL_RADIUS * 2, BALL_RADIUS * 2])
        self.image.set_colorkey(BLACK) # Torna o fundo do surface transparente
        pygame.draw.circle(self.image, BALL_COLOR, (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
        
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.centery = SCREEN_HEIGHT // 2
        
        # Velocidades iniciais (serão sobrescritas pelo reset_ball do Game)
        self.speed_x = BALL_SPEED_X_INITIAL
        self.speed_y = BALL_SPEED_Y_INITIAL

    def update(self):
        """
        Atualiza a posição da bola com base em seus vetores de velocidade.
        """
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

class Brick(pygame.sprite.Sprite):
    """
    Representa um tijolo destrutível no jogo.
    """

    def __init__(self, x, y, color, is_special=False):
        """
        Inicializa um tijolo.

        Args:
            x (int): Posição X superior esquerda.
            y (int): Posição Y superior esquerda.
            color (tuple): Cor RGB do tijolo.
            is_special (bool, optional): Se é um tijolo especial (ex: bônus). Padrão False.
        """
        super().__init__()
        self.is_special = is_special
        self.image = pygame.Surface([BRICK_WIDTH, BRICK_HEIGHT])
        
        if self.is_special:
            self.image.fill(YELLOW) # Destaque para tijolo especial
        else:
            self.image.fill(color)
            
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y