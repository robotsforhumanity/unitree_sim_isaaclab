from action_provider.action_provider_dds import DDSActionProvider
from action_provider.action_provider_trajectory import TrajectoryActionProvider,create_trajectory_generator
from action_provider.action_provider_policy import PolicyActionProvider
from action_provider.action_provider_replay import FileActionProviderReplay
from action_provider.action_provider_wholebody_dds import DDSWholebodyActionProvider
from pathlib import Path


def create_action_provider(env,args):
    """create action provider based on parameters"""
    if args.action_source == "dds":
        return DDSActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "dds_wholebody":
        return DDSWholebodyActionProvider(
            env=env,
            args_cli=args
        )
    elif args.action_source == "trajectory":
        trajectory_gen = create_trajectory_generator()
        return TrajectoryActionProvider(trajectory_gen)
    
    elif args.action_source == "policy":
        import threading
        import rclpy
        from action_provider.action_provider_groot import GrootActionProvider, ActionNode

        if not rclpy.ok():
            rclpy.init()
        action_node = ActionNode()

        # Crear un executor y correrlo en thread para no bloquear tu loop principal
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(action_node)
        threading.Thread(target=executor.spin, daemon=True).start()

        policy = GrootActionProvider(action_node)
        return policy

    elif args.action_source == "replay":
        return FileActionProviderReplay(env=env,args_cli=args)
    else:
        print(f"unknown action source: {args.action_source}")
        return None