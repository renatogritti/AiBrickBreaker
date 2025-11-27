"""
-----------------------------------------------------------------------
Arquivo: demo.py
Data: 27/11/2025
Versão: 1.0
Autor: Renato Gritti
Descrição:
    Script de demonstração do agente treinado. Roda em tempo real (60fps)
    e exibe as probabilidades de decisão do agente no console.
-----------------------------------------------------------------------
"""

import os
import torch
from stable_baselines3 import PPO
from src.rl_env import BrickBreakerEnv
from src.config import MODEL_PATH

def demo():
    """
    Carrega o modelo e executa o jogo em loop para demonstração.
    """
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"Modelo não encontrado em {MODEL_PATH}.zip. Por favor, execute 'python train.py' primeiro.")
        return

    print(f"Carregando modelo de {MODEL_PATH}...")
    
    # render_mode='human' garante 60 FPS
    env = BrickBreakerEnv(render_mode='human')
    
    model = PPO.load(MODEL_PATH, env=env)
    
    print("Iniciando demo... Pressione 'q' para sair.")
    
    obs, _ = env.reset()
    
    while True:
        # deterministic=False para manter comportamento exploratório/probabilístico do treino
        action, _states = model.predict(obs, deterministic=False)
        
        # Debug: Imprime probabilidades da rede neural
        with torch.no_grad():
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs[0].cpu().numpy()
            
        print(f"\rProbabilidades -> Ficar: {probs[0]:.2f} | Esq: {probs[1]:.2f} | Dir: {probs[2]:.2f}", end="")
        
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            obs, _ = env.reset()
            print("\nReiniciando jogo...")
            
        # Verifica se a janela foi fechada
        if not env.game.running:
            break
            
    env.close()
    print("\nDemo finalizada.")

if __name__ == "__main__":
    demo()