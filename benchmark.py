"""
-----------------------------------------------------------------------
Arquivo: benchmark.py
Data: 24/12/2025
Versão: 1.0
Autor: Renato Gritti (via Gemini Agent)
Descrição:
    Script de benchmark para avaliar performance do modelo e detectar
    viés direcional. Usado para comparar modelos antes/depois das melhorias.
-----------------------------------------------------------------------
"""

import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from src.rl_env import BrickBreakerEnv
from src.config import MODEL_PATH, LOGS_DIR

def benchmark_model(model_path, num_episodes=100, render=False):
    """
    Avalia o modelo em múltiplos episódios e coleta métricas.
    
    Args:
        model_path (str): Caminho para o modelo .zip
        num_episodes (int): Número de episódios para avaliar
        render (bool): Se True, renderiza o jogo (mais lento)
    
    Returns:
        dict: Dicionário com métricas coletadas
    """
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ Modelo não encontrado em {model_path}.zip")
        return None
    
    print(f"📊 Carregando modelo de {model_path}...")
    
    # Setup environment
    render_mode = 'human' if render else None
    env = DummyVecEnv([lambda: BrickBreakerEnv(render_mode=render_mode)])
    env = VecFrameStack(env, n_stack=4)
    
    # Load normalization stats if exist
    stats_path = os.path.join(LOGS_DIR, "vec_normalize.pkl")
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    
    model = DQN.load(model_path, env=env)
    
    # Métricas a coletar
    episode_rewards = []
    episode_lengths = []
    action_counts = {0: 0, 1: 0, 2: 0}  # 0=Stay, 1=Left, 2=Right
    level_2_reached = 0
    level_2_completed = 0
    
    print(f"🎮 Executando {num_episodes} episódios de benchmark...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        max_level_reached = 1
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Conta ações
            action_counts[int(action[0])] += 1
            
            episode_reward += reward[0]
            episode_length += 1
            
            # Tenta pegar nível atual (via get_attr)
            try:
                games = env.get_attr("game")
                if games:
                    current_level = games[0].level
                    max_level_reached = max(max_level_reached, current_level)
            except:
                pass
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if max_level_reached >= 2:
            level_2_reached += 1
        if max_level_reached >= 3:
            level_2_completed += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  Episódio {episode + 1}/{num_episodes} - Reward: {episode_reward:.1f}, Nível Max: {max_level_reached}")
    
    env.close()
    
    # Calcula estatísticas
    total_actions = sum(action_counts.values())
    action_distribution = {
        'stay': action_counts[0] / total_actions if total_actions > 0 else 0,
        'left': action_counts[1] / total_actions if total_actions > 0 else 0,
        'right': action_counts[2] / total_actions if total_actions > 0 else 0
    }
    
    # Calcula viés (ratio direita/esquerda)
    if action_counts[1] > 0:
        bias_ratio = action_counts[2] / action_counts[1]
    else:
        bias_ratio = float('inf') if action_counts[2] > 0 else 1.0
    
    metrics = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'level_2_success_rate': level_2_reached / num_episodes * 100,
        'level_2_completion_rate': level_2_completed / num_episodes * 100,
        'action_distribution': action_distribution,
        'bias_ratio': bias_ratio,
        'episode_rewards': episode_rewards
    }
    
    return metrics

def print_metrics(metrics, model_name="Modelo"):
    """Imprime métricas de forma formatada."""
    print(f"\n{'='*60}")
    print(f"📈 Resultados do Benchmark - {model_name}")
    print(f"{'='*60}")
    print(f"Reward Médio:              {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Steps Médios por Episódio: {metrics['avg_length']:.1f}")
    print(f"Taxa de Sucesso Nível 2:   {metrics['level_2_success_rate']:.1f}%")
    print(f"Taxa de Conclusão Nível 2: {metrics['level_2_completion_rate']:.1f}%")
    print(f"\n🎯 Distribuição de Ações:")
    print(f"  Parado:   {metrics['action_distribution']['stay']*100:.1f}%")
    print(f"  Esquerda: {metrics['action_distribution']['left']*100:.1f}%")
    print(f"  Direita:  {metrics['action_distribution']['right']*100:.1f}%")
    print(f"\n⚖️  Viés Direcional (Direita/Esquerda): {metrics['bias_ratio']:.2f}")
    
    # Análise do viés
    if 0.9 <= metrics['bias_ratio'] <= 1.1:
        print(f"   ✅ EXCELENTE: Sem viés significativo (alvo: 0.9-1.1)")
    elif 0.8 <= metrics['bias_ratio'] <= 1.2:
        print(f"   ⚠️  ACEITÁVEL: Viés leve (alvo: 0.9-1.1)")
    else:
        print(f"   ❌ PROBLEMA: Viés forte detectado! (alvo: 0.9-1.1)")
    
    print(f"{'='*60}\n")

def compare_models(old_model_path, new_model_path, num_episodes=100):
    """
    Compara dois modelos lado a lado.
    
    Args:
        old_model_path (str): Caminho para modelo antigo
        new_model_path (str): Caminho para modelo novo
        num_episodes (int): Número de episódios para cada modelo
    """
    print("\n" + "="*60)
    print("🔬 COMPARAÇÃO DE MODELOS")
    print("="*60)
    
    # Benchmark modelo antigo
    if os.path.exists(f"{old_model_path}.zip"):
        print("\n1️⃣  Avaliando modelo ANTIGO...")
        old_metrics = benchmark_model(old_model_path, num_episodes, render=False)
        if old_metrics:
            print_metrics(old_metrics, "Modelo Antigo")
    else:
        print(f"\n⚠️  Modelo antigo não encontrado em {old_model_path}.zip")
        old_metrics = None
    
    # Benchmark modelo novo
    if os.path.exists(f"{new_model_path}.zip"):
        print("\n2️⃣  Avaliando modelo NOVO...")
        new_metrics = benchmark_model(new_model_path, num_episodes, render=False)
        if new_metrics:
            print_metrics(new_metrics, "Modelo Novo")
    else:
        print(f"\n⚠️  Modelo novo não encontrado em {new_model_path}.zip")
        new_metrics = None
    
    # Comparação
    if old_metrics and new_metrics:
        print("\n" + "="*60)
        print("📊 COMPARAÇÃO DIRETA")
        print("="*60)
        
        improvement_reward = ((new_metrics['avg_reward'] - old_metrics['avg_reward']) / 
                             abs(old_metrics['avg_reward']) * 100) if old_metrics['avg_reward'] != 0 else 0
        improvement_level2 = new_metrics['level_2_success_rate'] - old_metrics['level_2_success_rate']
        
        print(f"Melhoria em Reward:         {improvement_reward:+.1f}%")
        print(f"Melhoria em Nível 2:        {improvement_level2:+.1f} pontos percentuais")
        print(f"Viés Antigo:                {old_metrics['bias_ratio']:.2f}")
        print(f"Viés Novo:                  {new_metrics['bias_ratio']:.2f}")
        
        bias_improvement = abs(1.0 - new_metrics['bias_ratio']) < abs(1.0 - old_metrics['bias_ratio'])
        print(f"Correção de Viés:           {'✅ SIM' if bias_improvement else '❌ NÃO'}")
        print("="*60 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark do modelo Brick Breaker AI')
    parser.add_argument('--model', type=str, default=MODEL_PATH, 
                       help='Caminho para o modelo (sem .zip)')
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Número de episódios para avaliar')
    parser.add_argument('--render', action='store_true', 
                       help='Renderizar o jogo durante benchmark')
    parser.add_argument('--compare', type=str, default=None,
                       help='Caminho para modelo antigo para comparação')
    
    args = parser.parse_args()
    
    if args.compare:
        # Modo comparação
        compare_models(args.compare, args.model, args.episodes)
    else:
        # Modo single
        metrics = benchmark_model(args.model, args.episodes, args.render)
        if metrics:
            print_metrics(metrics)
