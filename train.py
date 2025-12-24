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
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from src.rl_env import BrickBreakerEnv
from src.config import (
    MODEL_PATH, 
    LOGS_DIR, 
    TOTAL_TIMESTEPS, 
    LEARNING_RATE, 
    BUFFER_SIZE,
    LEARNING_STARTS,
    BATCH_SIZE,
    TAU,
    GAMMA,
    TRAIN_FREQ,
    GRADIENT_STEPS,
    TARGET_UPDATE_INTERVAL,
    EXPLORATION_FRACTION,
    EXPLORATION_INITIAL_EPS,
    EXPLORATION_FINAL_EPS,
    NET_ARCH
)

class KeyboardInterruptCallback(BaseCallback):
    """
    Callback para interromper o treinamento de forma segura ao pressionar 'q' no jogo.
    """
    def __init__(self, verbose=0):
        super(KeyboardInterruptCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Acessa a instância do jogo via get_attr (compatível com VecEnv Wrappers)
        try:
            games = self.training_env.get_attr("game")
            if games and not games[0].running:
                print("\nInterrupção detectada ('q' pressionado). Parando treinamento...")
                return False
        except Exception as e:
            # Em caso de erro ao acessar, não interrompe, apenas loga se verbose
            if self.verbose > 0:
                print(f"Erro ao verificar estado do jogo: {e}")
        return True

def train():
    """
    Configura e executa o loop de treinamento.
    """
    # Garante que diretórios existam
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Create vectorized environment for wrappers
    env = DummyVecEnv([lambda: BrickBreakerEnv()])
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Limpa modelo antigo se existir (para garantir nova arquitetura)
    if os.path.exists(f"{MODEL_PATH}.zip"):
        print("Removendo modelo antigo para iniciar novo treinamento...")
        os.remove(f"{MODEL_PATH}.zip")

    print("Criando novo modelo DQN com arquitetura personalizada...")
    # NOTA: Este é DQN Vanilla (Stable-Baselines3 não suporta Dueling DQN nativamente)
    # Para upgrade futuro, considerar QRDQN do sb3-contrib (Distributional RL)
    # Instalação: pip install sb3-contrib
    # Uso: from sb3_contrib import QRDQN
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_initial_eps=EXPLORATION_INITIAL_EPS,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
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
        
        # Salva estatísticas de normalização
        stats_path = os.path.join(LOGS_DIR, "vec_normalize.pkl")
        print(f"Salvando estatísticas de normalização em {stats_path}...")
        env.save(stats_path)
        
        env.close()
        print("Concluído.")

if __name__ == "__main__":
    train()
