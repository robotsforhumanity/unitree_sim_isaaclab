
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
parser.add_argument("--max_frames", type=int, default=1000, help="maximum number of synthetic frames to capture (default: 1000)")


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

def setup_signal_handlers(controller, dds_manager=None, simulation_app=None):
    """set signal handlers"""
    def signal_handler(signum, frame):
        global _shutdown_requested, _shutdown_count
        _shutdown_count += 1
        
        # If user presses Ctrl+C multiple times (3+), force exit immediately
        if _shutdown_count >= 3:
            print(f"\n⚠️  Forzando salida inmediata (Ctrl+C presionado {_shutdown_count} veces)...")
            print("⚠️  Los videos NO se generarán")
            
            # Close simulation_app first to prevent it from staying open
            if simulation_app is not None:
                try:
                    print("🔒 Cerrando Isaac Sim...")
                    simulation_app.close()
                    print("✓ Isaac Sim cerrado")
                except Exception as e:
                    print(f"⚠️  Error cerrando Isaac Sim: {e}")
            
            import os
            os._exit(1)
        
        print(f"\n🛑 Señal recibida ({signum}), deteniendo simulación... (intento {_shutdown_count}/3)")
        
        if _shutdown_count == 1:
            try:
                controller.stop()
                print("✓ Controller detenido")
            except Exception as e:
                print(f"⚠️  Error deteniendo controller: {e}")
            try:
                if dds_manager is not None:
                    dds_manager.stop_all_communication()
                    print("✓ DDS detenido")
            except Exception as e:
                print(f"⚠️  Error deteniendo DDS: {e}")
            
            # Set flag FIRST to stop data capture immediately
            _shutdown_requested = True
            print("🛑 DETENIENDO CAPTURA DE DATOS SINTÉTICOS...")
            
            # Stop simulation app to break the main loop
            if simulation_app is not None:
                try:
                    print("⏹️  Deteniendo Isaac Sim...")
                    simulation_app.close()
                    print("✓ Isaac Sim detenido")
                except Exception as e:
                    print(f"⚠️  Error deteniendo Isaac Sim: {e}")
            
            print("🎬 Preparando para generar videos al finalizar...")
            print("💡 Presiona Ctrl+C dos veces más para forzar salida sin generar videos")
        else:
            print(f"⚠️  Presiona Ctrl+C {3 - _shutdown_count} vez(ces) más para forzar salida")
    
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
    recording_shm = None
    recording_active = False  # Start with recording OFF
    frames_captured = 0  # Counter for captured frames
    max_frames = args_cli.max_frames  # Maximum frames to capture
    video_generated = False  # Flag to ensure video is only generated once
    
    if args_cli.record_synthetic:
        print("\n" + "="*80)
        print("🎥 INICIALIZANDO SISTEMA DE GRABACIÓN SINTÉTICA")
        print("="*80)
        print("⏳ Esto tomará ~10 segundos (solo una vez al inicio)")
        print(f"📊 Se capturarán automáticamente {max_frames} frames")
        print(f"📁 Datos se guardarán en: {output_path_synthetic}/clip_000")
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
            
            # Set output path to clip_000 directly
            output_path_synthetic = os.path.abspath(os.path.join(args_cli.generate_data_dir, "synthetic_replicator", "clip_000"))
            os.makedirs(output_path_synthetic, exist_ok=True)
            
            print("Initializing DiskBackend...")
            backend = rep.backends.get("DiskBackend")
            backend.initialize(output_dir=output_path_synthetic)
            
            print("Initializing CosmosWriter...")
            cosmos_writer = rep.writers.get("CosmosWriter")
            cosmos_writer.initialize(backend=backend, use_instance_id=True)
            
            # DON'T attach yet - will attach when loop starts
            print("\n" + "="*80)
            print("✅ SISTEMA DE GRABACIÓN LISTO")
            print("="*80)
            print(f"📁 Datos se guardarán en: {output_path_synthetic}")
            print("🎥 Capturará: RGB, Depth, Segmentación, Edges")
            print(f"📊 Límite: {max_frames} frames")
            print("⏸️  La grabación iniciará automáticamente con el simulador")
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
        
        # Attach render product and start recording if synthetic data is enabled
        if cosmos_writer and not recording_active:
            print("\n" + "="*80)
            print("🎬 INICIANDO GRABACIÓN SINTÉTICA")
            print("="*80)
            print("🔥🔥🔥 VERSION DEL CODIGO: 2025-12-01 20:43 🔥🔥🔥")  # MARKER PARA VERIFICAR VERSION
            try:
                cosmos_writer.attach(render_product_ref)
                recording_active = True
                print(f"✅ Grabación activa - capturando hasta {max_frames} frames")
                print("="*80 + "\n")
            except Exception as e:
                print(f"❌ Error adjuntando render_product: {e}")
                cosmos_writer = None

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
                    
                    # Read command from DDS for scene resets
                    reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                    
                    # Process DDS commands (scene resets only, synthetic recording is continuous)
                    if reset_pose_cmd is not None:
                        current_category = str(reset_pose_cmd.get("reset_category"))
                        
                        # Handle scene resets
                        if current_category == '1':
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

                # Check synthetic data recording progress
                # NOTE: The orchestrator runs automatically, we just monitor progress
                if cosmos_writer and recording_active and not _shutdown_requested and not video_generated:
                    # Use the REAL frame count from cosmos_writer
                    actual_frames = cosmos_writer._frame_id
                    
                    # Show progress periodically
                    if loop_count % 50 == 0 or actual_frames % 10 == 0:
                        print(f"📊 Progreso: {actual_frames}/{max_frames} frames ({(actual_frames/max_frames)*100:.1f}%)", flush=True)
                    
                    # Check if we reached the limit
                    if actual_frames >= max_frames: # Corre a 60 FPS
                        print("\n" + "="*80, flush=True)
                        print(f"🛑 LÍMITE ALCANZADO: {actual_frames}/{max_frames} frames", flush=True)
                        print("⏸️  Preparando para generar video...", flush=True)
                        print("="*80, flush=True)
                        recording_active = False  # Stop monitoring
                        frames_captured = actual_frames  # Update our counter
                        
                        # Generate video FIRST (before stopping orchestrator)
                        if not video_generated:
                            print("\n🎬 GENERANDO VIDEO...", flush=True)
                            print("="*80, flush=True)
                            try:
                                # Skip I/O wait - just give it a moment
                                print("💾 Esperando 5s para que termine I/O pendiente...", flush=True)
                                time.sleep(5)
                                print("✓ Espera completada", flush=True)
                                
                                print("\n📹 Generando video (puede tardar varios minutos)...", flush=True)
                                print("⚠️  Por favor espera, no cierres el simulador...", flush=True)
                                cosmos_writer.on_final_frame()
                                print("✓ on_final_frame() completado", flush=True)
                                video_generated = True
                                
                                print("\n" + "="*80, flush=True)
                                print("✅ ✅ ✅ VIDEO SINTÉTICO GENERADO EXITOSAMENTE ✅ ✅ ✅", flush=True)
                                print(f"📁 Ubicación: {output_path_synthetic}", flush=True)
                                print("="*80, flush=True)
                                
                                # NOW stop the orchestrator to prevent further captures
                                print("\n⏸️  Deteniendo orchestrator de Replicator...", flush=True)
                                try:
                                    import omni.replicator.core as rep
                                    
                                    # Check current state
                                    is_running_before = rep.orchestrator.get_is_started()
                                    print(f"   Estado ANTES: orchestrator.is_started = {is_running_before}", flush=True)
                                    
                                    # Stop it
                                    rep.orchestrator.stop()
                                    
                                    # Verify it stopped
                                    is_running_after = rep.orchestrator.get_is_started()
                                    print(f"   Estado DESPUÉS: orchestrator.is_started = {is_running_after}", flush=True)
                                    
                                    if not is_running_after:
                                        print("✅ Orchestrator DETENIDO exitosamente", flush=True)
                                    else:
                                        print("⚠️  Orchestrator parece seguir activo", flush=True)
                                        
                                except Exception as stop_error:
                                    print(f"❌ Error deteniendo orchestrator: {stop_error}", flush=True)
                                    import traceback
                                    traceback.print_exc()
                                
                                print("\n" + "="*80, flush=True)
                                print("ℹ️  El simulador continuará corriendo para teleoperar.", flush=True)
                                print("ℹ️  Los datos sintéticos YA NO se están capturando.", flush=True)
                                print("ℹ️  Presiona Ctrl+C cuando quieras salir.", flush=True)
                                print("="*80 + "\n", flush=True)
                                
                            except Exception as video_error:
                                print(f"\n❌ ❌ ❌ ERROR GENERANDO VIDEO: {video_error}", flush=True)
                                import traceback
                                traceback.print_exc()
                                print("⚠️  Intenta cerrar el simulador con Ctrl+C para intentar generar el video en el finally", flush=True)
                
                # Show a message every 5 seconds if video was generated but simulator continues
                elif video_generated and loop_count % 2500 == 0:  # ~5 seconds at 500Hz
                    print(f"[INFO] Simulador activo (loop {loop_count}) - Datos sintéticos YA NO se están capturando", flush=True)
                    

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
        # Generate video BEFORE program exits (only if not already generated)
        if cosmos_writer and not video_generated:
            print("\n" + "="*80)
            print("⏹️  FINALIZANDO GRABACIÓN SINTÉTICA")
            print("="*80)
            
            try:
                frame_count = frames_captured
                print(f"📊 Total de frames capturados: {frame_count}")
                
                if frame_count > 0:
                    # Ensure all pending I/O is flushed
                    print("💾 Esperando que termine la escritura de frames pendientes...")
                    
                    try:
                        # Import replicator
                        import omni.replicator.core as rep
                        
                        # Stop the orchestrator to prevent new captures
                        print("   1/2: Deteniendo orchestrator...")
                        rep.orchestrator.stop()
                        print("   ✓ Orchestrator detenido")
                        
                        # Wait for all pending I/O operations
                        print("   2/2: Esperando escritura de frames (puede tardar)...")
                        
                        # Try to access the I/O queue directly
                        try:
                            backend = rep.backends.get("DiskBackend")
                            if backend and hasattr(backend, '_io_queue'):
                                backend._io_queue.wait_until_done()
                                print("   ✓ Todos los frames escritos en disco")
                            else:
                                # Fallback: wait a reasonable time
                                print("   ⚠️  No se puede acceder a I/O queue, esperando 15s...")
                                time.sleep(15)
                                print("   ✓ Espera completada")
                        except AttributeError:
                            # If _io_queue doesn't exist, just wait
                            print("   ⚠️  Método de espera no disponible, esperando 15s...")
                            time.sleep(15)
                            print("   ✓ Espera completada")
                        
                    except Exception as flush_error:
                        print(f"⚠️  Advertencia durante flush: {flush_error}")
                        print("   Esperando 10s adicionales por seguridad...")
                        time.sleep(10)
                    
                    print("\n🎬 Generando video de la sesión completa...")
                    print("   Esto puede tardar varios minutos dependiendo del número de frames...")
                    print("   ⚠️  NO CIERRES ESTA VENTANA, el video se está generando...")
                    
                    cosmos_writer.on_final_frame()
                    video_generated = True
                    
                    print("\n✅ VIDEO GENERADO EXITOSAMENTE")
                    print(f"📁 Ubicación: {output_path_synthetic}")
                else:
                    print("⚠️  No se capturaron frames, no hay video para generar")
                    
                print(f"={'='*80}\n")
            except Exception as e:
                print(f"\n❌ ERROR GENERANDO VIDEO: {e}")
                import traceback
                traceback.print_exc()
                print("\n⚠️  El video NO se generó, pero las imágenes están guardadas")
        
        # Close the simulator
        print("\n🔒 Cerrando simulador...", flush=True)
        try:
            simulation_app.close()
            print("✅ Simulador cerrado exitosamente", flush=True)
        except Exception as close_error:
            print(f"⚠️  Error al cerrar simulador: {close_error}", flush=True)


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
