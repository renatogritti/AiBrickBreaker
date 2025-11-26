"""
Este módulo contém todas as configurações e constantes globais utilizadas no jogo Brick Breaker.
Isso inclui dimensões da tela, cores, e propriedades dos elementos do jogo como a raquete e a bola.
"""

# Dimensões da tela do jogo
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60 # Taxa de quadros por segundo

# Definições de Cores (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Configurações da Raquete
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
PADDLE_SPEED = 10 # Velocidade de movimento da raquete

# Configurações da Bola
BALL_RADIUS = 10
BALL_SPEED_X = 5 # Velocidade inicial da bola no eixo X
BALL_SPEED_Y = -5 # Velocidade inicial da bola no eixo Y (negativo para cima)
BALL_SPEED_INCREASE_PERCENTAGE = 0.05 # Percentual de aumento da velocidade da bola por nível
