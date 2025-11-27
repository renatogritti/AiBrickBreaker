"""
-----------------------------------------------------------------------
Arquivo: train.py
Data: 27/11/2025
Versão: 1.0
Autor: Renato Gritti
Descrição:
    Script principal para treinamento do agente PPO.
    Gerencia salvamento, carregamento e interrupção via teclado.
-----------------------------------------------------------------------
"""

import os
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.rl_env import BrickBreakerEnv
from src.config import (
    MODEL_PATH, 
    LOGS_DIR, 
    TOTAL_TIMESTEPS, 
    LEARNING_RATE, 
    ENTROPY_COEF, 
    GAMMA, 
    BATCH_SIZE, 
    NET_ARCH
)

class KeyboardInterruptCallback(BaseCallback):
    """
    Callback para interromper o treinamento de forma segura ao pressionar 'q' no jogo.
    """
    def __init__(self, verbose=0):
        super(KeyboardInterruptCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Acessa a instância do jogo dentro do ambiente envelopado
        game = self.training_env.envs[0].unwrapped.game
        
        if not game.running:
            print("\nInterrupção detectada ('q' pressionado). Parando treinamento...")
            return False
        return True

def train():
    """
    Configura e executa o loop de treinamento.
    """
    # Garante que diretórios existam
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    env = BrickBreakerEnv()
    
    # Limpa modelo antigo se existir (para garantir nova arquitetura)
    if os.path.exists(f"{MODEL_PATH}.zip"):
        print("Removendo modelo antigo para iniciar novo treinamento...")
        os.remove(f"{MODEL_PATH}.zip")

    print("Criando novo modelo PPO com arquitetura personalizada...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=LEARNING_RATE, 
        ent_coef=ENTROPY_COEF, 
        gamma=GAMMA, 
        batch_size=BATCH_SIZE,
        policy_kwargs=dict(net_arch=NET_ARCH),
        tensorboard_log=LOGS_DIR
    )

    callback = KeyboardInterruptCallback()

    print("Iniciando treinamento... Pressione 'q' na janela do jogo para salvar e sair.")
    print("Nota: O agente buscará ativamente a bola (Reward Shaping ativo).")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"Salvando modelo em {MODEL_PATH}...")
        model.save(MODEL_PATH)
        env.close()
        print("Concluído.")

if __name__ == "__main__":
    train()
