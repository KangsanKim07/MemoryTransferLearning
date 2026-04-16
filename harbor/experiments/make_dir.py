import os
import shutil
import argparse

def make_dir(run_id: str):
    if os.path.lexists("outputs/current"):
        os.unlink("outputs/current")
    os.makedirs(f"outputs/{run_id}", exist_ok=True)
    os.symlink(f"/home/ubuntu/harbor/outputs/{run_id}", "outputs/current")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args()
    make_dir(args.run_id)