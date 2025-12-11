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
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
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
    
    env = DummyVecEnv([lambda: BrickBreakerEnv(render_mode='human')])
    env = VecFrameStack(env, n_stack=4)
    
    # Carrega estatísticas se existirem
    from src.config import LOGS_DIR
    stats_path = os.path.join(LOGS_DIR, "vec_normalize.pkl")
    
    if os.path.exists(stats_path):
        print(f"Carregando estatísticas de normalização de {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        env.training = False # Importante: não atualizar estatísticas durante teste
        env.norm_reward = False
    else:
        print("Aviso: Estatísticas de normalização não encontradas. Usando padrão (sem normalização efetiva).")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    model = DQN.load(MODEL_PATH, env=env)
    
    print("Iniciando demo... Pressione 'q' para sair.")
    
    # VecEnv.reset() retorna apenas obs (sem info)
    obs = env.reset()
    
    while True:
        # deterministic=False para manter comportamento exploratório/probabilístico do treino
        action, _states = model.predict(obs, deterministic=False)
        
        # Debug: Imprime Q-Values da rede neural (DQN)
        with torch.no_grad():
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            q_values = model.q_net(obs_tensor)
            q_values = q_values.cpu().numpy()[0]
            
        print(f"\rQ-Values -> Ficar: {q_values[0]:.2f} | Esq: {q_values[1]:.2f} | Dir: {q_values[2]:.2f}", end="")
        
        obs, reward, done, info = env.step(action)
        
        # VecEnv reseta automaticamente quando done=True, então não precisamos chamar reset manualmente
        # Mas queremos saber se resetou para imprimir mensagem
        if hasattr(env, 'get_attr'):
             # Tenta pegar flag 'game_over' ou similar do jogo real se possível, mas VecEnv abstrai isso.
             # Verificamos 'terminal_observation' em info para saber se acabou
             if len(info) > 0 and 'terminal_observation' in info[0]:
                 print("\nReiniciando jogo...")

        # Verifica se a janela foi fechada acessando atributo do jogo
        try:
             games = env.get_attr("game")
             if games and not games[0].running:
                 break
        except:
             break
            
    env.close()
    print("\nDemo finalizada.")

if __name__ == "__main__":
    demo()