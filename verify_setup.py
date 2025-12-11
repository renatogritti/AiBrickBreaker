
import os
import shutil
from stable_baselines3 import DQN
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

def verify():
    print("Verifying environment setup...")
    
    # 1. Test Environment and Wrappers
    try:
        env = DummyVecEnv([lambda: BrickBreakerEnv()])
        env = VecFrameStack(env, n_stack=4)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        print("Integration successfully: Environment wrapped with FrameStack(4) and Normalize.")
        print(f"Observation Space shape: {env.observation_space.shape}")
    except Exception as e:
        print(f"FAILED to wrap environment: {e}")
        return

    # 2. Test Model Creation
    print("Creating DQN model with new architecture...")
    try:
        model = DQN(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=LEARNING_RATE,
            buffer_size=min(BUFFER_SIZE, 1000), # Reduce buffer for quick test
            learning_starts=100, # Reduce for quick test
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
        print(f"Model created successfully. Architecture: {NET_ARCH}")
    except Exception as e:
        print(f"FAILED to create model: {e}")
        return

    # 3. Test Training Step (Just 100 steps)
    print("Running short training session (100 steps)...")
    try:
        model.learn(total_timesteps=100)
        print("Training loop ran successfully.")
    except Exception as e:
        print(f"FAILED during training loop: {e}")
        return

    # 4. Test Saving Stats
    stats_path = os.path.join(LOGS_DIR, "vec_normalize_test.pkl")
    print(f"Testing stats saving to {stats_path}...")
    try:
        env.save(stats_path)
        if os.path.exists(stats_path):
            print("Stats file created successfully.")
        else:
             print("FAILED: Stats file not found after save.")
             return
    except Exception as e:
        print(f"FAILED to save stats: {e}")
        return

    # 5. Test Loading Stats
    print("Testing stats loading...")
    try:
         env2 = DummyVecEnv([lambda: BrickBreakerEnv()])
         env2 = VecFrameStack(env2, n_stack=4)
         env2 = VecNormalize.load(stats_path, env2)
         print("Stats loaded successfully.")
    except Exception as e:
         print(f"FAILED to load stats: {e}")
         return

    print("\nVERIFICATION PASSED!")

if __name__ == "__main__":
    verify()
