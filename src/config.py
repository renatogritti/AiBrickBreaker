"""
-----------------------------------------------------------------------
Arquivo: src/config.py
Data: 27/11/2025
Versão: 1.0
Autor: Renato Gritti
Descrição:
    Arquivo central de configuração do projeto. Contém todas as constantes
    globais, parâmetros de física, configurações de Reinforcement Learning
    e flags de sistema.
-----------------------------------------------------------------------
"""

import os

# =============================================================================
# Configurações de Sistema e Display
# =============================================================================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CAPTION = "Brick Breaker AI - v1.0"
FPS_HUMAN = 60       # Taxa de quadros para humanos (demo/main)
FPS_TRAIN = 0        # Taxa de quadros para treino (0 = ilimitado)

# Flag para habilitar/desabilitar som globalmente
ENABLE_SOUND = False 

# =============================================================================
# Definições de Cores (RGB)
# =============================================================================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# =============================================================================
# Configurações da Raquete (Paddle)
# =============================================================================
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
PADDLE_SPEED = 10
PADDLE_COLOR = WHITE
PADDLE_START_Y_OFFSET = 10 # Distância do fundo da tela

# =============================================================================
# Configurações da Bola (Ball)
# =============================================================================
BALL_RADIUS = 10
BALL_SPEED_X_INITIAL = 5   # Velocidade base horizontal
BALL_SPEED_Y_INITIAL = -5  # Velocidade base vertical (negativo sobe)
BALL_SPEED_INCREASE = 0.05 # % de aumento por nível
BALL_COLOR = WHITE

# Randomização Inicial da Bola
BALL_RANDOM_X_OFFSET = 20  # Variação +/- do centro da raquete
BALL_RANDOM_SPEED_MIN = 3.0
BALL_RANDOM_SPEED_MAX = 7.0

# =============================================================================
# Configurações de Tijolos (Bricks)
# =============================================================================
BRICK_WIDTH = 60
BRICK_HEIGHT = 20
BRICK_GAP = 10
BRICK_OFFSET_TOP = 50
BRICK_OFFSET_LEFT = 35
BRICK_COLORS = [RED, GREEN, BLUE]
SPECIAL_BRICK_CHANCE = 0.1 # 10% de chance a partir do nível 2

# =============================================================================
# Configurações de Reinforcement Learning (RL)
# =============================================================================
# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODEL_NAME = "ppo_brickbreaker"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# Hiperparâmetros de Treino
TOTAL_TIMESTEPS = 2_000_000
LEARNING_RATE = 0.0003
ENTROPY_COEF = 0.01
GAMMA = 0.99
BATCH_SIZE = 64
NET_ARCH = dict(pi=[256, 256], vf=[256, 256]) # Arquitetura da rede (Actor-Critic)

# Sistema de Recompensa (Reward Shaping)
REWARD_HIT_BRICK = 10       # Ganho ao quebrar tijolo
REWARD_HIT_PADDLE = 10      # Ganho ao rebater na raquete
REWARD_LOSE_LIFE = -50      # Penalidade ao perder vida
REWARD_TRACKING_FACTOR = 0.1 # Fator de recompensa por seguir a bola (shaping)
