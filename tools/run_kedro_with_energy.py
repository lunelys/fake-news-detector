import os
import subprocess
import sys
from datetime import datetime

from codecarbon import EmissionsTracker


def main():
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
    result = subprocess.call(
        [sys.executable, "-m", "kedro", "run", "--pipeline", "nlp_cleaning"],
        cwd=kedro_dir,
    )
    emissions = tracker.stop()

    print(f"Energy tracking complete. Emissions (kgCO2): {emissions}")
    sys.exit(result)


if __name__ == "__main__":
    main()
