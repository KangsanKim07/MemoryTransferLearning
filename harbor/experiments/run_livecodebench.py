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
import traceback


def run_iter(run_id: str, iter_idx: int, task_name: str):
    if os.path.exists(f"outputs/{run_id}/livecodebench/{iter_idx}/result.json"):
        return
    
    print(f"\033[93m[LiveCodeBench] Iteration {iter_idx} started for task {task_name}\033[0m")

    print(f"[LiveCodeBench] Running the agent for task {task_name}")
    try:
        proc = subprocess.run(
            [
                "harbor",
                "run",
                "--agent-import-path",
                "src.harbor.agents.installed.mini_swe_agent_sham:MiniSweAgentSham",
                "-o",
                f"outputs/{run_id}/livecodebench",
                "--job-name",
                str(iter_idx),
                "--task-name",
                task_name,
                "-n",
                "1",
                "-m",
                "openai/gpt-5-mini",
                # "openrouter/openai/gpt-oss-120b",
                "-p",
                "datasets/livecodebench/release_v6",
                "--timeout-multiplier",
                "2.0",
                "--ek",
                f"run_dir=/home/ubuntu/harbor/outputs/{run_id}",
                "--ak",
                "step_limit=50",
                "--no-delete",
                "--no-force-build"
            ],
            timeout=3600,
        )
    except Exception as e:
        print(f"[ERROR] harbor run for {task_name} failed: {e}", flush=True)
        print(f"[TIMEOUT] harbor run for {task_name} exceeded limit.", flush=True)
        return

    # log_dir = f"outputs/{run_id}/livecodebench/{iter_idx}"
    # print(f"[LiveCodeBench] LLM judgement is started for task {task_name}")
    # judgements, reward, trajectory, task, commands = run_llm_judge(task_name, log_dir, "livecodebench")
    # if judgements is None and reward is None and trajectory is None:
    #     print(f"[LiveCodeBench] No trajectory found for task {task_name}")
    #     return
    # print(f"[LiveCodeBench] Update util scores for task {task_name}")
    # update_util_scores(log_dir, "livecodebench", task_name, judgements)
    # print(f"[LiveCodeBench] Memory extraction is started for task {task_name}")
    # # extract_memory(judgement, trajectory, log_dir, task_name, task, "livecodebench")
    # if all(judgements):
    #     extract_workflow_memory(True, trajectory, log_dir, task_name, task, commands, "livecodebench")
    # insight_judge = True if sum(judgements) >= 2 else False
    # extract_insight_memory(insight_judge, trajectory, log_dir, task_name, task, commands, "livecodebench")
    # print(f"[LiveCodeBench] Memory extraction is completed for task {task_name}")
    return


def main(args):
    run_id = args.run_id
    prepare_output_dir(run_id, "livecodebench")
    for iter_idx, task_name in tqdm(enumerate(get_task_names("livecodebench")), total=len(get_task_names("livecodebench")), desc="Running tasks"):
        run_iter(run_id, iter_idx, task_name)


def _run_iter_wrapper(args):
    try:
        return run_iter(*args)
    except Exception as e:
        print(f"[ERROR] run_iter for {args[1]} failed: {e}", flush=True)
        traceback.print_exc()
        return


def run_parallel(args):
    run_id = args.run_id
    # prepare_output_dir(run_id, "livecodebench")
    if not os.path.exists(f"outputs/{run_id}"):# or not os.path.lexists(f"outputs/current"):
        raise ValueError(f"Output directory {run_id} does not exist")

    task_names = os.listdir("/home/ubuntu/harbor/datasets/livecodebench/release_v6")
    task_names = random.sample(task_names, 100)
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
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--num-proc", type=int, default=2)
    args = parser.parse_args()
    if args.parallel:
        run_parallel(args)
    else:
        main(args)