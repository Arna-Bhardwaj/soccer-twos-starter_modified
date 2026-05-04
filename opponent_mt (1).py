import os
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import ray
from ray import tune
from soccer_twos import EnvType
from utils import create_rllib_env

# Detect available CPUs (Handles both local machines and HPC SLURM clusters)
allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
# Reserve 1-2 cores for the OS and Ray Driver to prevent crashing
NUM_WORKERS = max(1, allocated_cpus - 2) 
NUM_ENVS_PER_WORKER = 1

if __name__ == "__main__":
    os.environ["RAY_DISABLE_DASHBOARD"] = "1"
    
    # Initialize Ray. If on a cluster, it will connect automatically.
    ray_address = os.environ.get("RAY_ADDRESS", None)
    if ray_address:
        ray.init(address=ray_address, include_dashboard=False)
    else:
        ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)

    # Get observation and action space dynamically
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    print(f"Starting training with {NUM_WORKERS} parallel CPU workers...")

    try:
        analysis = tune.run(
            "PPO",
            name="PPO_STAGE3_SELFPLAY",
            sync_to_driver=False,
            config={
                
                "num_gpus": 0,               
                "num_gpus_per_worker": 0,
                "num_workers": NUM_WORKERS,  
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                
                "log_level": "INFO",
                "framework": "torch",

                # Self-play Multi-Agent Setup
                "multiagent": {
                    "policies": {
                        "default": (None, obs_space, act_space, {}),
                    },
                    "policy_mapping_fn": tune.function(lambda _: "default"),
                    "policies_to_train": ["default"],
                },

                "env": "Soccer",
                "env_config": {
                    "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                    "variation": EnvType.multiagent_player,
                    "flatten_branched": True,
                    
                    "dispersion_threshold": 1.5, 
                },

                "model": {
                    "vf_share_layers": False,
                    "fcnet_hiddens": [512, 512, 256],
                    "fcnet_activation": "relu",
                    
                },

                # Batching 
                "rollout_fragment_length": 100,
                "train_batch_size": 12000,
                "sgd_minibatch_size": 2048,
                "num_sgd_iter": 10,
                "lr": 5e-5,
                "gamma": 0.995,
                "lambda": 0.95,
                "clip_param": 0.2,
                "vf_loss_coeff": 0.5,
                "entropy_coeff": 0.003,
            },
            stop={
                "timesteps_total": 15_000_000, # Increased slightly since it will train faster
            },
            checkpoint_freq=100,
            checkpoint_at_end=True,
            local_dir="./ray_results",
        )

        best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
        print(f"Best Trial: {best_trial}")
        
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial, metric="episode_reward_mean", mode="max"
        )
        print(f"Best Checkpoint: {best_checkpoint}")
        print("Done Stage 3 training")
        
    finally:
        ray.shutdown()