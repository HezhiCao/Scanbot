import argparse
from pathlib import Path
import shutil
import multiprocessing

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

    agent_config.sensor_specifications = [rgb_sensor_spec]
    return habitat_sim.Configuration(simulator_config, [agent_config])


def fill_navmesh_settings():
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.cell_height = 0.10
    navmesh_settings.agent_max_climb = 0.15
    return navmesh_settings

def generate_single_scene(scene_id, output_path=None):
    scene_path = Path(scene_id)
    if not scene_path.exists() or scene_path.suffix != ".glb":
        raise RuntimeError("Invalid scene glb file")

    if output_path is None:
        output_path = str(scene_path.with_suffix(".navmesh"))
        backup_path = Path(output_path + ".backup")
        if not backup_path.exists():
            Path(output_path).rename(backup_path)

    if not Path(output_path).exists():
        simulator_config = make_simulator_config(str(scene_id))
        sim = habitat_sim.Simulator(simulator_config)
        navmesh_success = sim.recompute_navmesh(sim.pathfinder, fill_navmesh_settings())

        if navmesh_success:
            sim.pathfinder.save_nav_mesh(output_path)
        else:
            raise RuntimeError("Failed to generate navmesh!")

        print("\n\033[0;32m\033[1mGenerating " + output_path + " ...\033[0m\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--scene_id", help="Which scene to recompute navmesh", type=str ,default=None,
    )
    parser.add_argument(
        "-o", "--output", help="Output file for generated navmesh", type=str, default=None,
    )
    parser.add_argument(
        "-d", "--dataset_dir", help="Directory path to whole scene dataset", type=str, default=None,
    )
    args = parser.parse_args()

    if args.dataset_dir is not None:
        assert args.scene_id is None, "Cannot specify both scene_id and dataset_dir"
        scenes = Path(args.dataset_dir).glob("**/*.glb")
        with multiprocessing.Pool(8) as pool:
            for _ in pool.imap_unordered(generate_single_scene, scenes):
                pass
    else:
        assert args.scene_id is not None, "Must specify one of scene_id or dataset_dir"
        generate_single_scene(args.scene_id, args.output)

if __name__ == "__main__":
    main()
