"""
-----------------------------------------------------------------------
Arquivo: src/rl_env.py
Data: 27/11/2025
Versão: 1.0
Autor: Renato (via Gemini Agent)
Descrição:
    Wrapper do Gymnasium para o jogo Brick Breaker. Converte o jogo em um
    ambiente compatível com bibliotecas de RL como Stable Baselines3.
-----------------------------------------------------------------------
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.game import Game
from src.config import FPS_HUMAN, FPS_TRAIN

class BrickBreakerEnv(gym.Env):
    """
    Ambiente Gymnasium personalizado para o Brick Breaker.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        """
        Inicializa o ambiente.

        Args:
            render_mode (str, optional): 'human' para renderizar em tempo real (60fps),
                                         None para velocidade máxima (treino).
        """
        super(BrickBreakerEnv, self).__init__()
        
        self.render_mode = render_mode
        self.game = Game()
        
        # Espaço de Ação Discreto: 0=Ficar, 1=Esquerda, 2=Direita
        self.action_space = spaces.Discrete(3)
        
        # Espaço de Observação: [Paddle X, Ball X, Ball Y, Ball Speed X, Ball Speed Y, Rel X, Paddle Speed]
        # Valores normalizados aproximados
        # PaddleX, BallX, BallY, BallVX, BallVY, RelX, PaddleVX
        low = np.array([0.0, 0.0, 0.0, -5.0, -5.0, -2.0, -2.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.2, 5.0, 5.0, 2.0, 2.0], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para um novo episódio.
        """
        super().reset(seed=seed)
        self.game.reset_game()
        return self.game.get_state(), {}

    def step(self, action):
        """
        Executa uma ação no ambiente.
        """
        # Define o FPS com base no modo de renderização
        fps = FPS_HUMAN if self.render_mode == 'human' else FPS_TRAIN
        
        # Executa passo no jogo
        obs, reward, done = self.game.step(action, fps=fps)
        
        truncated = False 
        info = {}
        
        return obs, reward, done, truncated, info

    def render(self):
        """
        Renderização é tratada internamente pela classe Game durante o step.
        """
        pass

    def close(self):
        """
        Fecha recursos se necessário.
        """
        pass