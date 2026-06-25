import os
import subprocess
import sys
import argparse
from datetime import datetime

from codecarbon import EmissionsTracker


def main():
    parser = argparse.ArgumentParser(description="Run a Kedro pipeline with CodeCarbon tracking.")
    parser.add_argument(
        "--pipeline",
        default="nlp_scoring",
        help="Kedro pipeline to run. Use nlp_full for a full retraining run.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="For nlp_scoring, stop after the incremental filter node. This reads data but does not write outputs.",
    )
    parser.add_argument("kedro_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    kedro_dir = os.path.join(project_root, "bluesky-pipeline")
    output_dir = os.path.join(kedro_dir, "data", "08_reporting")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tracker = EmissionsTracker(
        output_dir=output_dir,
        output_file=f"energy_report_{timestamp}.csv",
        measure_power_secs=10,
    )

    tracker.start()
    kedro_args = args.kedro_args
    if kedro_args and kedro_args[0] == "--":
        kedro_args = kedro_args[1:]

    command = [sys.executable, "-m", "kedro", "run", "--pipeline", args.pipeline]
    if args.estimate_only:
        if args.pipeline != "nlp_scoring":
            print("--estimate-only is only supported with --pipeline nlp_scoring.")
            tracker.stop()
            sys.exit(2)
        command.extend(["--to-nodes", "filter_posts_for_incremental_scoring_node"])
    command.extend(kedro_args)

    print(f"Running: {' '.join(command)}")
    result = subprocess.call(command, cwd=kedro_dir)
    emissions = tracker.stop()

    print(f"Energy tracking complete. Emissions (kgCO2): {emissions}")
    sys.exit(result)


if __name__ == "__main__":
    main()
