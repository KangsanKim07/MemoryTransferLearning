import json
import argparse
import subprocess
import os
import random
random.seed(42)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import multiprocessing
from experiments.utils.utils import *
from experiments.utils.memory_extract import *


def run_iter(run_id: str, iter_idx: int, task_name: str):
    if os.path.exists(f"outputs/{run_id}/spider2-dbt/{iter_idx}/result.json"):
        return
    
    print(f"\033[93m[Spider2-DBT] Iteration {iter_idx} started for task {task_name}\033[0m")

    print(f"[Spider2-DBT] Running the agent for task {task_name}")
    try:
        proc = subprocess.run(
            [
                "harbor",
                "run",
                "--agent-import-path",
                "src.harbor.agents.installed.mini_swe_agent_sham:MiniSweAgentSham",
                "-o",
                f"outputs/{run_id}/spider2-dbt",
                "--job-name",
                str(iter_idx),
                "--task-name",
                task_name,
                "-n",
                "1",
                "-m",
                "openai/gpt-5-mini",
                "-d",
                "spider2-dbt@1.0",
                "--timeout-multiplier",
                "3.0",
                "--ek",
                f"run_dir=/home/ubuntu/harbor/outputs/{run_id}",
                # "--no-delete",
                # "--no-force-build"
            ],
            timeout=3600,
        )
    except Exception as e:
        print(f"[ERROR] harbor run for {task_name} failed: {e}", flush=True)
        print(f"[TIMEOUT] harbor run for {task_name} exceeded limit.", flush=True)
        return

    # log_dir = f"outputs/{run_id}/autocodebench/{iter_idx}"
    # print(f"[AutoCodeBench] LLM judgement is started for task {task_name}")
    # judgement, reward, trajectory, task, commands = run_llm_judge(task_name, log_dir, "autocodebench")
    # print(f"[AutoCodeBench] Memory extraction is started for task {task_name}")
    # if judgement is None and reward is None and trajectory is None:
    #     print(f"[AutoCodeBench] No trajectory found for task {task_name}")
    #     return
    # extract_memory(judgement, trajectory, log_dir, task_name, task, "autocodebench")
    # if judgement:
    # extract_workflow_memory(judgement, trajectory, log_dir, task_name, task, commands, "autocodebench")
    # extract_summary_memory(judgement, trajectory, log_dir, task_name, task, commands, "autocodebench")
    # extract_insight_memory(judgement, trajectory, log_dir, task_name, task, commands, "autocodebench")
    # print(f"[AutoCodeBench] Memory extraction is completed for task {task_name}")
    return

def _run_iter_wrapper(args):
    try:
        return run_iter(*args)
    except Exception as e:
        print(f"[ERROR] run_iter for {args[1]} failed: {e}", flush=True)
        return


def run_parallel(args):
    run_id = args.run_id
    if not os.path.exists(f"outputs/{run_id}"):# or not os.path.lexists(f"outputs/current"):
        raise ValueError(f"Output directory {run_id} does not exist")

    task_names = get_task_names("spider2-dbt")
    task_names = sorted(task_names)
    # task_names = random.sample(task_names, 100)
    num_tasks = len(task_names)

    inputs = [(run_id, idx, name) for idx, name in enumerate(task_names)]

    with multiprocessing.Pool(processes=args.num_proc) as pool:
        for _ in tqdm(
            pool.imap(_run_iter_wrapper, inputs, chunksize=1),
            total=num_tasks,
            desc="Running tasks"
        ):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--num-proc", type=int, default=2)
    args = parser.parse_args()
    if args.parallel:
        run_parallel(args)