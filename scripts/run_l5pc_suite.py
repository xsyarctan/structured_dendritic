from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standard L5PC experiment suite across multiple seeds.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "l5pc/cnn7",
            "l5pc/factorized_1layer",
            "l5pc/factorized_2layer",
            "l5pc/matched_residual_s4d",
        ],
        help="Hydra experiment presets to execute.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111, 2222, 3333], help="Seeds to run.")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra Hydra overrides to append.")
    args = parser.parse_args()

    for experiment in args.experiments:
        base_name = experiment.replace("/", "_")
        for seed in args.seeds:
            run_name = f"{base_name}_seed{seed}"
            command = [
                sys.executable,
                "train.py",
                f"experiment={experiment}",
                f"run.seed={seed}",
                f"run.name={run_name}",
                *args.extra,
            ]
            print(" ".join(command))
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
