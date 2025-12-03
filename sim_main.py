
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
#!/usr/bin/env python3
# main.py
import os

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

import argparse
import contextlib
import time
import sys
import signal
import torch
import gymnasium as gym
from pathlib import Path
import threading



# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

from image_server.image_server import ImageServer
from dds.dds_create import create_dds_objects,create_dds_objects_replay
# add command line arguments
parser = argparse.ArgumentParser(description="Unitree Simulation")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G129-Head-Waist-Fix", help="task name")
parser.add_argument("--action_source", type=str, default="dds", 
                   choices=["dds", "file", "trajectory", "policy", "replay","dds_wholebody"], 
                   help="Action source")


parser.add_argument("--robot_type", type=str, default="g129", help="robot type")
parser.add_argument("--enable_dex1_dds", action="store_true", help="enable gripper DDS")
parser.add_argument("--enable_dex3_dds", action="store_true", help="enable dexterous hand DDS")
parser.add_argument("--enable_inspire_dds", action="store_true", help="enable inspire hand DDS")
parser.add_argument("--stats_interval", type=float, default=10.0, help="statistics print interval (seconds)")

parser.add_argument("--file_path", type=str, default="xr_teleoperate/teleop/utils/data", help="file path (when action_source=file)")
parser.add_argument("--generate_data_dir", type=str, default="./data", help="save data dir")
parser.add_argument("--rerun_log", action="store_true", default=False, help="rerun log")
parser.add_argument("--replay_data",  action="store_true", default=False, help="replay data")

parser.add_argument("--modify_light",  action="store_true", default=False, help="modify light")
parser.add_argument("--modify_camera",  action="store_true", default=False,    help="modify camera")

# performance analysis parameters
parser.add_argument("--step_hz", type=int, default=500, help="control frequency")
parser.add_argument("--enable_profiling", action="store_true", default=True, help="enable performance analysis")
parser.add_argument("--profile_interval", type=int, default=500, help="performance analysis report interval (steps)")
parser.add_argument("--record_synthetic", action="store_true", default=False, help="enable synthetic data recording with Replicator")
parser.add_argument("--max_frames", type=int, default=0, help="maximum frames per episode (0=unlimited, controlled by STOP_REC)")


# add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


if args_cli.enable_dex3_dds and args_cli.enable_dex1_dds and args_cli.enable_inspire_dds:
    print("Error: enable_dex3_dds and enable_dex1_dds and enable_inspire_dds cannot be enabled at the same time")
    print("Please select one of the options")
    sys.exit(1)


import pinocchio 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from layeredcontrol.robot_control_system import (
    RobotController, 
    ControlConfig,
)

from dds.reset_pose_dds import *
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from tools.augmentation_utils import (
    update_light,
    batch_augment_cameras_by_name,
)

from tools.data_json_load import sim_state_to_json
from dds.sim_state_dds import *
from action_provider.create_action_provider import create_action_provider

# Import Replicator core
# Moved inside main or after app launch to ensure extensions are loaded
# import omni.replicator.core as rep
# from tools.cosmos_writer import CosmosWriter
# from omni.isaac.core.utils.semantics import add_update_semantics

# Global flag for graceful shutdown
_shutdown_requested = False
_shutdown_count = 0
_simulation_app_ref = None  # Global reference to close simulator on Ctrl+C

def setup_signal_handlers(controller, dds_manager=None, simulation_app=None):
    """set signal handlers"""
    global _simulation_app_ref
    _simulation_app_ref = simulation_app
    
    def signal_handler(signum, frame):
        global _shutdown_requested, _shutdown_count, _simulation_app_ref
        _shutdown_count += 1
        
        print(f"\n🛑 Ctrl+C recibido (intento {_shutdown_count}/2)", flush=True)
        
        # Second Ctrl+C: Force exit immediately
        if _shutdown_count >= 2:
            print("⚠️  Forzando salida inmediata...", flush=True)
            # Close simulator first
            if _simulation_app_ref is not None:
                try:
                    print("🔒 Cerrando Isaac Sim...", flush=True)
                    _simulation_app_ref.close()
                except:
                    pass
            import os
            os._exit(0)
        
        # First Ctrl+C: Graceful shutdown
        _shutdown_requested = True
        print("🔄 Cerrando gracefully... (presiona Ctrl+C de nuevo para forzar)", flush=True)
        
        try:
            controller.stop()
            print("✓ Controller detenido", flush=True)
        except Exception as e:
            print(f"⚠️  Error deteniendo controller: {e}", flush=True)
        
        try:
            if dds_manager is not None:
                dds_manager.stop_all_communication()
                print("✓ DDS detenido", flush=True)
        except Exception as e:
            print(f"⚠️  Error deteniendo DDS: {e}", flush=True)
        
        # Close simulator
        if _simulation_app_ref is not None:
            try:
                print("🔒 Cerrando Isaac Sim...", flush=True)
                _simulation_app_ref.close()
                print("✓ Isaac Sim cerrado", flush=True)
            except Exception as e:
                print(f"⚠️  Error cerrando Isaac Sim: {e}", flush=True)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """main function"""
    import os
    import atexit
    import time
    try:
        os.setpgrp()
        current_pgid = os.getpgrp()
        print(f"Setting process group: {current_pgid}")

        def cleanup_process_group():
            try:
                print(f"Cleaning up process group: {current_pgid}")
                import signal
                os.killpg(current_pgid, signal.SIGTERM)
            except Exception as e:
                print(f"Failed to clean up process group: {e}")

        atexit.register(cleanup_process_group)

    except Exception as e:
        print(f"Failed to set process group: {e}")

    print("=" * 60)
    print("robot control system started")
    print(f"Task: {args_cli.task}")
    print(f"Action source: {args_cli.action_source}")
    print("=" * 60)

    # parse environment configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
    except Exception as e:
        print(f"Failed to parse environment configuration: {e}")
        return
    
    # create environment
    print("\ncreate environment...")
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        print(f"\ncreate environment success ...")
    except Exception as e:
        print(f"\nFailed to create environment: {e}")
        return
    
    print("\n")
    print("***  Please left-click on the Sim window to activate rendering. ***")
    print("\n")
    # reset environment
    if args_cli.modify_light:
        update_light(
            prim_path="/World/light",
            color=(1.0, 0.8, 0.6),
            intensity=20000.0,
            position=(1.0, 2.0, 3.0),
            radius=0.1,
            enabled=True,
            cast_shadows=True
        )
    if args_cli.modify_camera:
        batch_augment_cameras_by_name(
            names=["front_cam"],
            focal_length=3.0,
            horizontal_aperture=22.0,
            vertical_aperture=16.0,
            exposure=0.8,
            focus_distance=1.2
        )
    env.sim.reset()
    env.reset()
    
    # Initialize CosmosWriter variables
    cosmos_writer = None
    render_product_ref = None  
    output_path_synthetic = None
    recording_active = False  # Start with recording OFF - will start when START_REC received
    frames_captured = 0  # Counter for captured frames
    max_frames = args_cli.max_frames  # Maximum frames to capture (0 = unlimited, controlled by STOP_REC)
    video_generated = False  # Flag to ensure video is only generated once per episode
    episode_id = 0  # Episode counter for synchronized recording
    last_recording_cmd = None  # Track last command to avoid duplicates
    
    # Initialize shared memory for recording commands (fallback to DDS)
    recording_shm = None
    last_shm_timestamp = 0.0  # Track last processed shared memory command timestamp
    if args_cli.record_synthetic:
        try:
            from tools.recording_shm import RecordingCommandShm
            recording_shm = RecordingCommandShm(is_writer=False)  # Simulator is the reader
            print("✅ Shared memory para comandos de grabación inicializada (reader)")
        except Exception as shm_err:
            print(f"⚠️  No se pudo inicializar shared memory: {shm_err}")
            recording_shm = None
    
    if args_cli.record_synthetic:
        print("\n" + "="*80)
        print("🎥 INICIALIZANDO SISTEMA DE GRABACIÓN SINTÉTICA SINCRONIZADA")
        print("="*80)
        print("⏳ Esto tomará ~10 segundos (solo una vez al inicio)")
        print("🔄 MODO SINCRONIZADO: La grabación se controla desde teleoperación")
        print("   - Presiona 's' en teleoperación para INICIAR grabación")
        print("   - Presiona 's' de nuevo para DETENER y generar video")
        if max_frames > 0:
            print(f"📊 Límite máximo por episodio: {max_frames} frames")
        else:
            print("📊 Sin límite de frames (controlado por STOP_REC)")
        print("="*80 + "\n")
        
        try:
            print("Importing omni.replicator.core...")
            import omni.replicator.core as rep
            print("Importing omni.usd...")
            import omni.usd
            print("Importing CosmosWriter...")
            from tools.cosmos_writer import CosmosWriter
            from pxr import Semantics
            
            def add_update_semantics(prim, label):
                if not prim.HasAPI(Semantics.SemanticsAPI):
                    sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                else:
                    sem = Semantics.SemanticsAPI.Get(prim, "Semantics")
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set("class")
                sem.GetSemanticDataAttr().Set(label)
            
            # Apply semantics
            print("Applying semantics...")
            objects_to_label = {
                "/World/envs/env_0/Robot": "Robot",
                "/World/envs/env_0/valvulaJoint": "Valvula",
                "/World/envs/env_0/Room/Robot_room": "Room",
                "/World/envs/env_0/Room/tgn": "Cartel",
                "/World/envs/env_0/Room/table": "Mesa",
            }
            stage = omni.usd.get_context().get_stage()
            for prim_path, label in objects_to_label.items():
                try:
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        add_update_semantics(prim, label)
                        print(f"  ✓ Labeled: {label}")
                except Exception as e:
                    print(f"  ⚠️ Could not label {prim_path}: {e}")
            
            # Create render product
            camera_path = "/World/envs/env_0/Robot/d435_link/front_cam"
            image_size = (1280, 800)
            print(f"\nCreating render product for {camera_path}...")
            render_product_ref = rep.create.render_product(camera_path, image_size)
            print("  ✓ Render product created")
            
            # Register and initialize writer
            print("\nRegistering CosmosWriter...")
            rep.WriterRegistry.register(CosmosWriter)
            
            # Set base output path (episodes will be created as subdirectories)
            output_path_synthetic = os.path.abspath(os.path.join(args_cli.generate_data_dir, "synthetic_replicator"))
            os.makedirs(output_path_synthetic, exist_ok=True)
            
            # Check for existing episodes to continue numbering
            existing_episodes = [d for d in os.listdir(output_path_synthetic) 
                               if d.startswith("episode_") and os.path.isdir(os.path.join(output_path_synthetic, d))]
            if existing_episodes:
                episode_id = max(int(ep.split("_")[1]) for ep in existing_episodes)
                print(f"📂 Encontrados {len(existing_episodes)} episodios existentes, continuando desde {episode_id + 1}")
            
            print("Initializing DiskBackend...")
            backend = rep.backends.get("DiskBackend")
            backend.initialize(output_dir=output_path_synthetic)
            
            print("Initializing CosmosWriter...")
            cosmos_writer = rep.writers.get("CosmosWriter")
            cosmos_writer.initialize(backend=backend, use_instance_id=True)
            
            # DON'T attach yet - will attach when START_REC is received
            print("\n" + "="*80)
            print("✅ SISTEMA DE GRABACIÓN SINCRONIZADA LISTO")
            print("="*80)
            print(f"📁 Base de datos: {output_path_synthetic}")
            print("🎥 Capturará: RGB, Depth, Segmentación, Edges")
            print("⏸️  ESPERANDO comando START_REC desde teleoperación...")
            print("   (Presiona 's' en teleoperación para iniciar grabación)")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n❌ ERROR initializing CosmosWriter: {e}")
            import traceback
            traceback.print_exc()
            cosmos_writer = None
            render_product_ref = None
        
        # Note: Recording continuously to clip_000, no episode management needed

    
    # create simplified control configuration
    try:    
        control_config = ControlConfig(
            step_hz=args_cli.step_hz,
            replay_mode=args_cli.replay_data
        )
    except Exception as e:
        print(f"Failed to create control configuration: {e}")
        sys.exit(1)
    
    # create controller

    if not args_cli.replay_data:
        print("========= create image server =========")
        try:
            server = ImageServer(fps=30, Unit_Test=False)
        except Exception as e:
            print(f"Failed to create image server: {e}")
            sys.exit(1)
        print("========= create image server success =========")
        print("========= create dds =========")
        try:
            reset_pose_dds,sim_state_dds,dds_manager = create_dds_objects(args_cli,env)
        except Exception as e:
            print(f"Failed to create dds: {e}")
            sys.exit(1)
        print("========= create dds success =========")
    else:
        print("========= create dds =========")
        try:
            create_dds_objects_replay(args_cli,env)
        except Exception as e:
            print(f"Failed to create dds: {e}")
            sys.exit(1)
        print("========= create dds success =========")
        from tools.data_json_load import get_data_json_list
        print("========= get data json list =========")
        data_idx=0
        data_json_list = get_data_json_list(args_cli.file_path)
        if args_cli.action_source != "replay":
            args_cli.action_source = "replay"
        print("========= get data json list success =========")
    # create action provider
    
    print(f"\ncreate action provider: {args_cli.action_source}...")
    try:
        action_provider = create_action_provider(env,args_cli)
        if action_provider is None:
            print("action provider creation failed, exiting")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to create action provider: {e}")
        sys.exit(1)
    
    # set action provider
    print("========= create controller =========")
    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)
    print("========= create controller success =========")
    
    # configure performance analysis
    if args_cli.enable_profiling:
        controller.set_profiling(True, args_cli.profile_interval)
        print(f"performance analysis enabled, report every {args_cli.profile_interval} steps")
    else:
        controller.set_profiling(False)
        print("performance analysis disabled")


    # set signal handlers
    if not args_cli.replay_data:
        setup_signal_handlers(controller, dds_manager, simulation_app)
    else:
        setup_signal_handlers(controller, None, simulation_app)
    print("Note: The DDS in Sim transmits messages on channel 1. Please ensure that other DDS instances use the same channel for message exchange by setting: ChannelFactoryInitialize(1).")
    try:
        # start controller - start asynchronous components
        print("========= start controller =========")
        controller.start()
        print("========= start controller success =========")
        
        # Clear DDS commands NOW (after DDS is initialized)
        if args_cli.record_synthetic and not args_cli.replay_data:
            try:
                print("Clearing previous DDS commands...")
                reset_pose_dds.write_reset_pose_command("-1")
                print("DDS commands cleared")
            except Exception as e:
                print(f"Warning: Could not clear DDS commands: {e}")
        
        # main loop - execute in main thread to support rendering
        last_stats_time = time.time()
        loop_start_time = time.time()
        loop_count = 0
        last_loop_time = time.time()
        recent_loop_times = []  # for calculating moving average frequency
        
        # Recording will start when START_REC command is received from teleoperator
        # DO NOT auto-attach - wait for synchronized command
        print("\n" + "="*80)
        print("📊 ESTADO DEL SISTEMA DE GRABACIÓN")
        print("="*80)
        print(f"   cosmos_writer: {cosmos_writer is not None}")
        print(f"   recording_shm: {recording_shm is not None}")
        print(f"   record_synthetic flag: {args_cli.record_synthetic}")
        if cosmos_writer:
            print("🔄 MODO SINCRONIZADO ACTIVO")
            print("   Presiona 's' en teleoperación para iniciar grabación")
        else:
            print("⚠️  cosmos_writer NO inicializado!")
            print("   Asegúrate de pasar --record_synthetic al iniciar")
        print("="*80 + "\n")

        # use torch.inference_mode() - removed KeyboardInterrupt suppression for graceful shutdown
        with torch.inference_mode():
            while simulation_app.is_running() and controller.is_running:
                # Check if shutdown was requested (Ctrl+C)
                global _shutdown_requested
                if _shutdown_requested:
                    print("\n⏹️  Shutdown solicitado, finalizando loop principal...")
                    break
                
                current_time = time.time()
                loop_count += 1
                
                # DEBUG: Print EVERY loop (first 50 loops)
                if loop_count <= 50:
                    print(f"[LOOP {loop_count}] START", flush=True)
                
                if not args_cli.replay_data:
                    env_state = env.scene.get_state()
                    env_state_json =  sim_state_to_json(env_state)
                    sim_state = {"init_state":env_state_json,"task_name":args_cli.task}
                    sim_state_dds.write_sim_state_data(sim_state)
                    
                    # Read command from DDS for scene resets AND recording control
                    reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                    current_category = None
                    
                    # First try DDS
                    if reset_pose_cmd is not None:
                        dds_category = str(reset_pose_cmd.get("reset_category"))
                        # Debug: show DDS commands that are not -1
                        if dds_category not in ['-1', 'None', '', None] and loop_count % 100 == 0:
                            print(f"📡 [DDS] Comando recibido: '{dds_category}'", flush=True)
                        current_category = dds_category
                    
                    # Fallback: Check shared memory for recording commands
                    if recording_shm is not None:
                        shm_cmd = recording_shm.read_command()
                        # Debug: show what we read from shared memory periodically
                        if loop_count % 500 == 0 and shm_cmd:
                            print(f"📡 [SHM DEBUG] Raw: '{shm_cmd}' | DDS: '{current_category}'", flush=True)
                        
                        if shm_cmd and shm_cmd.strip() and current_category in [None, '-1', 'None', '']:
                            # Format from teleop: "COMMAND|timestamp"
                            shm_parts = shm_cmd.split('|')
                            if len(shm_parts) >= 2:
                                shm_category = shm_parts[0].strip()
                                try:
                                    shm_timestamp = float(shm_parts[1])
                                    # Only process if this is a NEW command (timestamp > last processed)
                                    if shm_category in ['START_REC', 'STOP_REC'] and shm_timestamp > last_shm_timestamp:
                                        current_category = shm_category
                                        last_shm_timestamp = shm_timestamp
                                        print(f"📡 [SHM] Nuevo comando: '{shm_category}' (ts={shm_timestamp:.3f})", flush=True)
                                except (ValueError, IndexError) as e:
                                    if loop_count % 500 == 0:
                                        print(f"⚠️  [SHM] Error parseando: {e}", flush=True)
                    
                    # Process commands
                    if current_category is not None:
                        # Debug: Show received commands periodically
                        if loop_count % 500 == 0 and current_category not in ['-1', 'None', '']:
                            print(f"📡 [DEBUG] Comando actual: '{current_category}' | Último: '{last_recording_cmd}' | recording={recording_active}", flush=True)
                        
                        # Only process meaningful commands (ignore -1 which is the "cleared" state)
                        # Process if it's a new command different from last one
                        if current_category not in ['-1', 'None', ''] and current_category != last_recording_cmd:
                            print(f"📡 [CMD] Procesando comando: '{current_category}'", flush=True)
                            last_recording_cmd = current_category
                            
                            # Handle START_REC - Start synchronized recording
                            if current_category == 'START_REC':
                                print(f"📡 [START_REC] cosmos_writer={cosmos_writer is not None}, recording_active={recording_active}", flush=True)
                            
                            if current_category == 'START_REC' and cosmos_writer and not recording_active:
                                episode_id += 1
                                episode_dir = f"episode_{episode_id:04d}"
                                episode_path = os.path.join(output_path_synthetic, episode_dir)
                                os.makedirs(episode_path, exist_ok=True)
                                
                                print("\n" + "="*80)
                                print(f"🎬 INICIANDO GRABACIÓN - EPISODIO {episode_id}")
                                print("="*80)
                                print(f"📁 Guardando en: {episode_path}")
                                
                                try:
                                    # Set episode output directory
                                    cosmos_writer.set_episode_output(episode_dir)
                                    cosmos_writer._frame_id = 0  # Reset frame counter
                                    cosmos_writer._clip_idx = 0  # Reset clip index
                                    cosmos_writer.attach(render_product_ref)
                                    recording_active = True
                                    frames_captured = 0
                                    video_generated = False
                                    print(f"✅ Grabación ACTIVA")
                                    if max_frames > 0:
                                        print(f"📊 Límite: {max_frames} frames")
                                    else:
                                        print("📊 Sin límite de frames (detener con 's')")
                                    print("="*80 + "\n")
                                except Exception as e:
                                    print(f"❌ Error iniciando grabación: {e}")
                                    import traceback
                                    traceback.print_exc()
                                
                                reset_pose_dds.write_reset_pose_command(-1)
                            
                            # Handle STOP_REC - Stop recording and generate video
                            elif current_category == 'STOP_REC':
                                print(f"📡 [STOP_REC] cosmos_writer={cosmos_writer is not None}, recording_active={recording_active}", flush=True)
                                should_stop = cosmos_writer and recording_active
                                if should_stop:
                                    print("\n" + "="*80, flush=True)
                                    print(f"⏹️  DETENIENDO GRABACIÓN - EPISODIO {episode_id}", flush=True)
                                    print("="*80, flush=True)
                                    
                                    # Mark as not recording FIRST to stop new frames
                                    recording_active = False
                                    frames_captured = cosmos_writer._frame_id
                                    current_episode_id = episode_id  # Save for video generation
                                    current_episode_path = f"{output_path_synthetic}/episode_{episode_id:04d}"
                                    
                                    print(f"📊 Frames capturados: {frames_captured}", flush=True)
                                    print(f"📁 Episodio guardado en: {current_episode_path}", flush=True)
                                    
                                    # Generate video in background thread to not block simulation
                                    if frames_captured > 0:
                                        def generate_video_async(writer, ep_id, ep_path, frame_count):
                                            try:
                                                print(f"\n🎬 [Thread] Generando video episodio {ep_id}...", flush=True)
                                                print(f"   ⚠️  Esto puede tardar, por favor espera...", flush=True)
                                                
                                                # Wait for pending I/O
                                                time.sleep(2)
                                                
                                                writer.on_final_frame()
                                                
                                                print(f"\n✅ VIDEO GENERADO - EPISODIO {ep_id}", flush=True)
                                                print(f"📁 Ubicación: {ep_path}", flush=True)
                                                print(f"📊 Total frames: {frame_count}", flush=True)
                                            except Exception as ve:
                                                print(f"❌ Error generando video: {ve}", flush=True)
                                                import traceback
                                                traceback.print_exc()
                                        
                                        video_thread = threading.Thread(
                                            target=generate_video_async,
                                            args=(cosmos_writer, current_episode_id, current_episode_path, frames_captured),
                                            daemon=True
                                        )
                                        video_thread.start()
                                        print("🔄 Generación de video iniciada en background", flush=True)
                                    else:
                                        print("⚠️  No se capturaron frames, no hay video", flush=True)
                                    
                                    print("\n⏸️  Listo para siguiente episodio", flush=True)
                                    print("   Presiona 's' en teleoperación para iniciar", flush=True)
                                    print("="*80 + "\n", flush=True)
                                    
                                    reset_pose_dds.write_reset_pose_command(-1)
                                else:
                                    print(f"⚠️  [STOP_REC] Ignorado - recording_active={recording_active}", flush=True)
                            
                            # Handle scene resets
                            elif current_category == '1':
                                print("reset object")
                                env_cfg.event_manager.trigger("reset_object_self", env)
                                reset_pose_dds.write_reset_pose_command(-1)
                            elif current_category == '2':
                                print("reset all")
                                env_cfg.event_manager.trigger("reset_all_self", env)
                                reset_pose_dds.write_reset_pose_command(-1)
                else:
                    if action_provider.get_start_loop() and data_idx<len(data_json_list):
                        print(f"data_idx: {data_idx}")
                        sim_state,task_name = action_provider.load_data(data_json_list[data_idx])
                        if task_name!=args_cli.task:
                            raise ValueError(f" The {task_name} in the dataset is different from the {args_cli.task} being executed .")
                        env.reset_to(sim_state, torch.tensor([0], device=env.device), is_relative=True)
                        env.sim.reset()
                        time.sleep(1)
                        action_provider.start_replay()
                        data_idx+=1
                # print(f"env_state: {env_state}")
                # calculate instantaneous loop time
                loop_dt = current_time - last_loop_time
                last_loop_time = current_time
                recent_loop_times.append(loop_dt)
                
                # keep recent 100 loop times
                if len(recent_loop_times) > 100:
                    recent_loop_times.pop(0)
                
                # DEBUG: Before controller.step
                if loop_count <= 50:
                    print(f"[LOOP {loop_count}] Before controller.step()", flush=True)
                
                # execute control step (in main thread, support rendering)
                controller.step()

                # DEBUG: After controller.step
                if loop_count <= 50:
                    print(f"[LOOP {loop_count}] After controller.step()", flush=True)
                    print(f"[DEBUG] cosmos_writer={cosmos_writer is not None}, recording_active={recording_active}, shutdown={_shutdown_requested}", flush=True)

                # Check synthetic data recording progress (synchronized mode)
                if cosmos_writer and recording_active and not _shutdown_requested:
                    # Use the REAL frame count from cosmos_writer
                    actual_frames = cosmos_writer._frame_id
                    
                    # Show progress periodically (every ~1 second at 500Hz loop)
                    if loop_count % 500 == 0:
                        if max_frames > 0:
                            print(f"🎥 [Episodio {episode_id}] Grabando: {actual_frames}/{max_frames} frames ({(actual_frames/max_frames)*100:.1f}%)", flush=True)
                        else:
                            print(f"🎥 [Episodio {episode_id}] Grabando: {actual_frames} frames", flush=True)
                    
                    # Check if we reached the limit (only if max_frames > 0)
                    if max_frames > 0 and actual_frames >= max_frames:
                        print("\n" + "="*80, flush=True)
                        print(f"🛑 LÍMITE ALCANZADO - EPISODIO {episode_id}", flush=True)
                        print(f"📊 Frames: {actual_frames}/{max_frames}", flush=True)
                        print("="*80, flush=True)
                        
                        recording_active = False
                        frames_captured = actual_frames
                        
                        try:
                            # Stop the orchestrator
                            import omni.replicator.core as rep
                            try:
                                rep.orchestrator.stop()
                                print("✓ Orchestrator detenido", flush=True)
                            except Exception as orch_err:
                                print(f"⚠️  Error deteniendo orchestrator: {orch_err}", flush=True)
                            
                            # Generate video
                            print("\n🎬 Generando video...", flush=True)
                            time.sleep(3)
                            cosmos_writer.on_final_frame()
                            video_generated = True
                            
                            print(f"\n✅ VIDEO GENERADO - EPISODIO {episode_id}", flush=True)
                            print(f"📁 Ubicación: {output_path_synthetic}/episode_{episode_id:04d}", flush=True)
                            print("\n⏸️  Listo para siguiente episodio", flush=True)
                            print("   Presiona 's' en teleoperación para iniciar", flush=True)
                            print("="*80 + "\n", flush=True)
                            
                        except Exception as video_error:
                            print(f"\n❌ ERROR: {video_error}", flush=True)
                            import traceback
                            traceback.print_exc()
                
                # Periodic status when not recording
                elif cosmos_writer and not recording_active and loop_count % 5000 == 0:  # Every ~10 seconds
                    print(f"⏸️  [Sim activo] Esperando comando de grabación... (Episodios completados: {episode_id})", flush=True)
                    

                # print statistics and loop frequency periodically
                if current_time - last_stats_time >= args_cli.stats_interval:
                    # calculate while loop execution frequency
                    elapsed_time = current_time - loop_start_time
                    loop_frequency = loop_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # calculate moving average frequency (based on recent loop times)
                    if recent_loop_times:
                        avg_loop_time = sum(recent_loop_times) / len(recent_loop_times)
                        moving_avg_frequency = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
                        min_loop_time = min(recent_loop_times)
                        max_loop_time = max(recent_loop_times)
                        max_freq = 1.0 / min_loop_time if min_loop_time > 0 else 0
                        min_freq = 1.0 / max_loop_time if max_loop_time > 0 else 0
                    else:
                        moving_avg_frequency = 0
                        min_freq = max_freq = 0
                    
                    print(f"\n=== While loop execution frequency statistics ===")
                    print(f"loop execution count: {loop_count}")
                    print(f"running time: {elapsed_time:.2f} seconds")
                    print(f"overall average frequency: {loop_frequency:.2f} Hz")
                    print(f"moving average frequency: {moving_avg_frequency:.2f} Hz (last {len(recent_loop_times)} times)")
                    print(f"frequency range: {min_freq:.2f} - {max_freq:.2f} Hz")
                    print(f"average loop time: {(elapsed_time/loop_count*1000):.2f} ms")
                    if recent_loop_times:
                        print(f"recent loop time: {(avg_loop_time*1000):.2f} ms")
                    print(f"=============================")
                    
                    # print_stats(controller)
                    last_stats_time = current_time
       
                # check environment state
                if env.sim.is_stopped():
                    print("\nenvironment stopped")
                    break
                # rate_limiter.sleep(env)
        
        # Check if we exited due to shutdown request
        if _shutdown_requested:
            print("\n" + "="*80)
            print("⚠️  SIMULACIÓN INTERRUMPIDA POR USUARIO (Ctrl+C)")
            print("="*80)
            
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("⚠️  SIMULACIÓN INTERRUMPIDA POR USUARIO (Ctrl+C)")
        print("="*80)
    
    except Exception as e:
        print(f"\nprogram exception: {e}")
    
    finally:
        # Generate video BEFORE program exits if recording was active
        if cosmos_writer and recording_active:
            print("\n" + "="*80)
            print(f"⏹️  FINALIZANDO GRABACIÓN - EPISODIO {episode_id}")
            print("="*80)
            
            try:
                frame_count = cosmos_writer._frame_id
                print(f"📊 Frames capturados en episodio actual: {frame_count}")
                
                if frame_count > 0:
                    print("💾 Esperando que termine la escritura...")
                    
                    try:
                        # Import replicator
                        import omni.replicator.core as rep
                        
                        # Stop the orchestrator
                        print("   Deteniendo orchestrator...")
                        rep.orchestrator.stop()
                        print("   ✓ Orchestrator detenido")
                        
                        # Wait for I/O
                        time.sleep(5)
                        
                    except Exception as flush_error:
                        print(f"⚠️  Advertencia durante flush: {flush_error}")
                        time.sleep(5)
                    
                    print("\n🎬 Generando video del episodio...")
                    cosmos_writer.on_final_frame()
                    
                    print(f"\n✅ VIDEO GENERADO - EPISODIO {episode_id}")
                    print(f"📁 Ubicación: {output_path_synthetic}/episode_{episode_id:04d}")
                else:
                    print("⚠️  No se capturaron frames en este episodio")
                    
                print("="*80 + "\n")
            except Exception as e:
                print(f"\n❌ ERROR GENERANDO VIDEO: {e}")
                import traceback
                traceback.print_exc()
        
        elif cosmos_writer and episode_id > 0:
            print("\n" + "="*80)
            print(f"📊 RESUMEN DE SESIÓN")
            print("="*80)
            print(f"   Episodios completados: {episode_id}")
            print(f"   Ubicación: {output_path_synthetic}")
            print("="*80 + "\n")
        
        # Close shared memory
        if recording_shm is not None:
            try:
                recording_shm.close()
                print("✅ Shared memory cerrada")
            except Exception as shm_close_err:
                print(f"⚠️  Error cerrando shared memory: {shm_close_err}")
        
        # Close the simulator
        print("\n🔒 Cerrando simulador...", flush=True)
        try:
            simulation_app.close()
            print("✅ Simulador cerrado exitosamente", flush=True)
        except Exception as close_error:
            print(f"⚠️  Error al cerrar simulador: {close_error}", flush=True)
        
        # Force exit to terminate any remaining threads
        print("\n👋 Programa finalizado.", flush=True)
        import os
        os._exit(0)


# ============================================================================
# Entry point - Call main function
# ============================================================================
if __name__ == "__main__":
    main()


# ============================================================================
# Example commands
# ============================================================================
# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-Cylinder-G129-Dex1-Joint    --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129

# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-RedBlock-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex3-Joint     --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Inspire-Joint     --enable_inspire_dds --robot_type g129
