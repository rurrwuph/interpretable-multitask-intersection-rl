import glob
import os
import shutil
import subprocess
import numpy as np
import torch

# Flow & TraCI imports
from flow.core.params import VehicleParams, NetParams, SumoParams, EnvParams, InitialConfig, TrafficLightParams
from flow.envs import TestEnv

# Custom Project Modules
from intersection_netw import UnsignalizedIntersectionNetwork, SCENARIO_CONFIGS
from intersection_env import MultiTaskIntersectionEnv
from multitask_dqn_model import MultiTaskDQN

# File and Directory Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# TARGET_CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "---")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "scenario_i_frames")
VIDEO_NAME = os.path.join(SCRIPT_DIR, "scenario_i_demo.mp4")


def resolve_checkpoint(run_dir):
    """Finds the model weight file inside the target checkpoint folder."""
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Directory not found: {run_dir}")

    priority_files = [
        os.path.join(run_dir, "ckpt_final.pt"),
        os.path.join(run_dir, "best_model.pt"),
        os.path.join(run_dir, "model.pt"),
        os.path.join(run_dir, "model.pth"),
    ]
    for p in priority_files:
        if os.path.isfile(p):
            return p

    found = glob.glob(os.path.join(run_dir, "**/*.pt"), recursive=True) + \
            glob.glob(os.path.join(run_dir, "**/*.pth"), recursive=True)

    if not found:
        raise FileNotFoundError(f"No .pt or .pth weight files found inside: {run_dir}")

    found.sort(key=os.path.getmtime, reverse=True)
    return found[0]


def record_scenario_i_demo():
    TASK_CHOICE = "straight"
    VIEW_ID = "View #0"
    FPS = 10

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load exact MultiTaskDQN Model & Weights
    ckpt_path = resolve_checkpoint(TARGET_CKPT_DIR)
    print(f"\n[+] Loading checkpoint from: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MultiTaskDQN().to(device)

    checkpoint_data = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint_data, dict):
        if "model_state_dict" in checkpoint_data:
            policy.load_state_dict(checkpoint_data["model_state_dict"])
        elif "state_dict" in checkpoint_data:
            policy.load_state_dict(checkpoint_data["state_dict"])
        elif "model" in checkpoint_data:
            policy.load_state_dict(checkpoint_data["model"])
        else:
            policy.load_state_dict(checkpoint_data)
    else:
        policy.load_state_dict(checkpoint_data)
    policy.eval()

    # 2. Build Scenario i Geometry & Flow Environment
    scenario_i_cfg = SCENARIO_CONFIGS["scenario_i"]

    sim_params = SumoParams(
        sim_step=0.1,
        render=True,
        restart_instance=True
    )

    # Define vehicle types:
    # color="green" ensures SUMO spawns the rl vehicle as green from frame 0
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="rl",
        num_vehicles=0,
        color="green"
    )
    vehicles.add(
        veh_id="human",
        num_vehicles=25,
        color="white"
    )

    net_params = NetParams(additional_params=scenario_i_cfg)
    valid_spawn_edges = ["north_in", "south_in", "east_in", "west_in"]

    flow_network = UnsignalizedIntersectionNetwork(
        name="scenario_i_network",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=InitialConfig(
            edges_distribution=valid_spawn_edges
        ),
        traffic_lights=TrafficLightParams()
    )

    base_flow_env = TestEnv(
        env_params=EnvParams(horizon=500),
        sim_params=sim_params,
        network=flow_network
    )

    env = MultiTaskIntersectionEnv(base_flow_env)

    # 3. Reset Environment & Initialize Visuals
    print(f"[+] Initializing Scenario i (Task: '{TASK_CHOICE}')...")
    obs, info = env.reset(options={"task": TASK_CHOICE})

    traci_api = env.flow_env.k.kernel_api
    ego_id = env._current_ego_id

    # Center camera on the ego car immediately
    traci_api.gui.setZoom(VIEW_ID, 450)
    if ego_id in traci_api.vehicle.getIDList():
        traci_api.gui.trackVehicle(VIEW_ID, ego_id)

    step = 0
    done = False
    task_g_tensor = torch.tensor(env.active_g, dtype=torch.float32, device=device).unsqueeze(0)

    print("[+] Recording simulation frames...")
    try:
        while not done and step < 400:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Model Forward Pass & Action Selection via masked_q
            with torch.no_grad():
                r_tensor = policy(obs_tensor)
                q_values = MultiTaskDQN.masked_q(r_tensor, task_g_tensor)
                action = torch.argmax(q_values, dim=1).item()

            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # Capture frame screenshot
            frame_path = os.path.join(OUTPUT_DIR, f"frame_{step:05d}.png")
            traci_api.gui.screenshot(VIEW_ID, frame_path)
            step += 1

        print(f"[+] Simulation ended at step {step}.")
        print(f"Outcome: Success={step_info.get('is_success')}, Collision={step_info.get('is_collision')}")

    finally:
        env.flow_env.terminate()

    # 4. Compile Video via FFmpeg
    print("[+] Stitching frames into MP4 video...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-r", str(FPS),
        "-i", os.path.join(OUTPUT_DIR, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        VIDEO_NAME
    ]
    res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0:
        print(f"\n[✓] Demo video saved successfully:\n    {VIDEO_NAME}\n")
    else:
        print("\nFFmpeg error:")
        print(res.stderr)


if __name__ == "__main__":
    record_scenario_i_demo()