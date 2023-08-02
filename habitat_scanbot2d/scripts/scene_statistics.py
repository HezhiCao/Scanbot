import argparse
from typing import Dict, Any
from pathlib import Path
from concurrent import futures
from contextlib import closing

import habitat_sim


def make_simulator_config(scene_id: str):
    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = scene_id

    agent_config = habitat_sim.agent.AgentConfiguration()
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [640, 480]
    rgb_sensor_spec.position = [0.0, 1.25, 0.0]

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [640, 480]
    depth_sensor_spec.position = [0.0, 1.25, 0.0]

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    depth_sensor_spec.resolution = [640, 480]
    depth_sensor_spec.position = [0.0, 1.25, 0.0]

    agent_config.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    return habitat_sim.Configuration(simulator_config, [agent_config])


def gather_navigable_area_and_regions(scene_id) -> Dict[str, Any]:
    scene_path = Path(scene_id)
    if not scene_path.exists() or scene_path.suffix != ".glb":
        raise RuntimeError("Invalid scene glb file")

    stats = {}
    stats["scene_id"] = scene_path.stem
    simulator_config = make_simulator_config(str(scene_id))
    with closing(habitat_sim.Simulator(simulator_config)) as sim:
        stats["area"] = sim.pathfinder.navigable_area
        stats["level"] = len(sim.semantic_scene.levels)
        stats["region"] = 0
        for level in sim.semantic_scene.levels:
            stats["region"] += len(level.regions)

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scene_id",
        help="Which scene to gather statistics",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file statistics",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dataset_dir",
        help="Directory path to whole scene dataset",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.dataset_dir is not None:
        assert args.scene_id is None, "Cannot specify both scene_id and dataset_dir"
        scenes = Path(args.dataset_dir).glob("**/*.glb")
        output_lines = []
        with futures.ProcessPoolExecutor(max_workers=8) as executor:
            for stats in executor.map(gather_navigable_area_and_regions, scenes):
                output_line = " ".join(f"{k}: {v}" for k, v in stats.items()) + '\n'
                if args.output is not None:
                    output_lines.append(output_line)
                else:
                    print(output_line)

        if args.output is not None:
            with open(args.output, "wt") as fd:
                fd.writelines(output_lines)
    else:
        assert args.scene_id is not None, "Must specify one of scene_id or dataset_dir"
        stats = gather_navigable_area_and_regions(args.scene_id)
        for k, v in stats.items():
            print(k, v)


if __name__ == "__main__":
    main()
