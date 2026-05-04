from random import uniform as randfloat
import numpy as np
import gym
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """
    def __init__(self, env, dispersion_threshold=1.5):
        super().__init__(env)
        self.env = env
        self.dispersion_threshold = dispersion_threshold

    def step(self, action):
        if isinstance(action, dict):
            converted = {}
            for k, v in action.items():
                if hasattr(v, '__len__'):
                    converted[k] = int(v[0]) * 9 + int(v[1]) * 3 + int(v[2])
                else:
                    converted[k] = int(v)
            action = converted

        obs, rewards, dones, infos = self.env.step(action)
        rewards = self._shape_rewards(rewards, infos)
        return obs, rewards, dones, infos

    def _shape_rewards(self, rewards, infos):
        shaped = {}
        teammate_map = {0: 1, 1: 0, 2: 3, 3: 2}

        for agent_id, reward in rewards.items():
            bonus = 0.0

            # logging variables
            dist_bonus = 0.0
            vel_bonus = 0.0
            disp_penalty = 0.0

            info = infos.get(agent_id, {})
            teammate_id = teammate_map.get(agent_id)
            teammate_info = infos.get(teammate_id, {})

            if "ball_info" in info and "player_info" in info:
                ball_pos = np.array(info["ball_info"].get("position", [0.0, 0.0]))
                player_pos = np.array(info["player_info"].get("position", [0.0, 0.0]))

                # 1. Smooth Distance Shaping — reduced 10x from original
                dist = np.linalg.norm(ball_pos - player_pos)
                dist_bonus = 0.0001 * np.exp(-0.5 * dist)
                bonus += dist_bonus

                # 2. Velocity bonus — reduced 10x, gated by proximity
                gate = np.exp(-dist)
                ball_vel = info["ball_info"].get("velocity", [0.0, 0.0])
                attack_direction = 1.0 if agent_id in [0, 1] else -1.0
                velocity_towards_goal = ball_vel[0] * attack_direction

                if velocity_towards_goal > 0:
                    
                    clipped_vel = min(velocity_towards_goal, 15.0)
                    vel_bonus = gate * 0.00005 * clipped_vel
                    bonus += vel_bonus

                

            
            if "player_info" in info and "player_info" in teammate_info:
                p1_pos = np.array(info["player_info"].get("position", [0.0, 0.0]))
                p2_pos = np.array(teammate_info["player_info"].get("position", [0.0, 0.0]))
                team_dist = np.linalg.norm(p1_pos - p2_pos)
                if team_dist < self.dispersion_threshold:
                    disp_penalty = -0.0005  
                    bonus += disp_penalty

            # 5. Final reward
            shaped[agent_id] = reward + bonus

            # 6. Log components for TensorBoard
            if isinstance(info, dict):
                info["shaping_dist_bonus"] = dist_bonus
                info["shaping_vel_bonus"] = vel_bonus
                info["shaping_disp_penalty"] = disp_penalty
                info["shaping_total_bonus"] = bonus

        return shaped


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )

    # extract dispersion_threshold before passing to soccer_twos.make
    # soccer_twos.make does not accept this parameter
    dispersion_threshold = env_config.pop("dispersion_threshold", 1.5)

    env = soccer_twos.make(**env_config)

    if "multiagent" in env_config and not env_config["multiagent"]:
        return env

    return RLLibWrapper(env, dispersion_threshold=dispersion_threshold)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s