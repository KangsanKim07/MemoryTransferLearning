import os
import json
import pickle
from filelock import FileLock, Timeout
from openai import OpenAI
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.prompts.memory import *

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MEMORY_LOCK_TIMEOUT = 180

def parse_memory_items(text: str):
    items = []
    blocks = re.split(r'(?=^# Memory Item\s+\d+)', text, flags=re.MULTILINE)

    for block in blocks:
        if not block.strip():
            continue
        m_title = re.search(r'^## Title\s*(.+)', block, flags=re.MULTILINE)
        m_desc = re.search(r'^## Description\s*(.+)', block, flags=re.MULTILINE)
        m_content = re.search(
            r'^## Content\s*((?:.|\n)*?)(?=\n## |\Z)',
            block,
            flags=re.MULTILINE
        )

        if m_title and m_desc:
            title = m_title.group(1).strip()
            description = m_desc.group(1).strip()
            content = m_content.group(1).strip() if m_content else ""

            items.append({
                "title": title.strip(":").strip(),
                "description": description.strip(":").strip(),
                "content": content.strip(":").strip(),
            })

    return items


def parse_local_memory_items(text: str):
    data = json.loads(text)
    return data


def extract_rawtraj_memory(judgement: bool, trajectory: list, log_dir: str, task_name: str, task: str, commands: list, benchmark: str):
    new_memory = {
        "task_name": task_name,
        "task": task,
        "commands": commands,
        "benchmark": benchmark,
        "judgement": judgement,
        "type": "trajectory"
    }
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=new_memory["task"],
    )
    emb_vec = embedding.data[0].embedding
    new_memory["embedding"] = emb_vec

    try:
        memory_path = f"{log_dir.rsplit('/', 1)[0]}/trajectory_memory.pkl"
        lock_path = memory_path + ".lock"
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append(new_memory)
            new_text_memory = new_memory.copy()
            new_text_memory.pop("embedding")
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
            # with open(f"{log_dir}/trajectory_memory.json", "w") as f:
            #     json.dump({"memory": new_text_memory, "all_memory": [{"task_name": x["task_name"], "task": x["task"], "benchmark": x["benchmark"], "commands": x["commands"], "judgement": x["judgement"]} for x in all_memory]}, f, indent=4)
    except Timeout:
        print(
            f"[{benchmark}] Trajectory Memory lock timeout "
            f"(>{MEMORY_LOCK_TIMEOUT}s). Skipping trajectory memory update for {log_dir}."
        )
        return
    return


def extract_workflow_memory(judgement: bool, trajectory: list, log_dir: str, task_name: str, task: str, commands: list, benchmark: str):
    prompt = WORKFLOW_CORRECT_EXTRACT_PROMPT if judgement else WORKFLOW_WRONG_EXTRACT_PROMPT
    content = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"### Trajectory:\n{trajectory[1:]}"},
    ]
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=content,
        timeout=300.0,
        service_tier="flex",
    )
    memory_text = response.choices[0].message.content
    memory_path = f"{log_dir.rsplit('/', 1)[0]}/workflow_memory.pkl"
    lock_path = memory_path + ".lock"
    try:
        new_memory = parse_local_memory_items(memory_text)
    except:
        print(f"[{benchmark}] Failed to parse workflow memory items for {log_dir}")
        return
    if not isinstance(new_memory, dict):
        print(f"[{benchmark}] Failed to parse workflow memory items for {log_dir}")
        return
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=new_memory["goal"],
    )
    emb_vec = embedding.data[0].embedding
    new_elem = {
        "benchmark": benchmark,
        "task_name": task_name,
        "llm_judge": judgement,
        "task": task,
        "type": "workflow",
        "workflow": new_memory,
        "key_embedding": emb_vec
    }
    try:
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append(new_elem)
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
            # with open(f"{log_dir}/workflow_memory.json", "w") as f:
            #     json.dump({"memory": new_text_memory, "all_memory": [{"task_query": x["task_query"], "workflow": x["workflow"], "benchmark": x["benchmark"], "task_name": x["task_name"], "task": x["task"]} for x in all_memory]}, f, indent=4)
    except Timeout:
        print(
            f"[{benchmark}] Workflow Memory lock timeout "
            f"(>{MEMORY_LOCK_TIMEOUT}s). Skipping workflow memory update for {log_dir}."
        )
        return
    return

    
def extract_traj_memory(judgement: bool, trajectory: list, log_dir: str, task_name: str, task: str, commands: list, benchmark: str):
    system_prompt = CODE_SPECIFIC_CORRECT_PROMPT if judgement else CODE_SPECIFIC_WRONG_PROMPT
    content = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Trajectory:\n{trajectory}"},
    ]
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=content,
        timeout=300.0,
    )
    memory_text = response.choices[0].message.content or ""
    memory_path = f"{log_dir.rsplit('/', 1)[0]}/local_memory.pkl"
    lock_path = memory_path + ".lock"
    try:
        new_memory = parse_local_memory_items(memory_text)
        new_memory['task_query'] = task
    except:
        print(f"[{benchmark}] Failed to parse local memory items for {log_dir}")
        return
    if not isinstance(new_memory, dict):
        print(f"[{benchmark}] Failed to parse local memory items for {log_dir}")
        return
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=new_memory["generalized_query"],
    )
    emb_vec = embedding.data[0].embedding
    new_memory["generalized_query_embedding"] = emb_vec
    new_memory["benchmark"] = benchmark
    new_memory["task_name"] = task_name
    new_memory["commands"] = commands
    try:
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append(new_memory)
            new_text_memory = new_memory.copy()
            new_text_memory.pop("generalized_query_embedding")
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
            with open(f"{log_dir}/local_memory.json", "w") as f:
                json.dump({"memory": new_text_memory, "all_memory": [{"when_to_use": x["when_to_use"], "task_query": x["task_query"], "generalized_query": x["generalized_query"], "experience": x["experience"], "tags": x["tags"], "benchmark": x["benchmark"], "task_name": x["task_name"]} for x in all_memory[:-1]]}, f, indent=4)
    except Timeout:
        print(
            f"[{benchmark}] Local Memory lock timeout "
            f"(>{MEMORY_LOCK_TIMEOUT}s). Skipping local memory update for {log_dir}."
        )
        return


def extract_summary_memory(judgement: bool, trajectory: list, log_dir: str, task_name: str, task: str, commands: list, benchmark: str):
    prompt = SUMMARY_CORRECT_EXTRACT_PROMPT if judgement else SUMMARY_WRONG_EXTRACT_PROMPT
    content = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"### Trajectory:\n{trajectory[1:]}"},
    ]
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=content,
        timeout=300.0,
        service_tier="flex",
    )
    memory_text = response.choices[0].message.content
    memory_path = f"{log_dir.rsplit('/', 1)[0]}/summary_memory_{benchmark}.pkl"
    lock_path = memory_path + ".lock"
    try:
        new_memory = parse_local_memory_items(memory_text)
    except:
        print(f"[{benchmark}] Failed to parse workflow memory items for {log_dir}")
        return
    if not isinstance(new_memory, dict):
        print(f"[{benchmark}] Failed to parse workflow memory items for {log_dir}")
        return
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=new_memory["task_summary"],
    )
    emb_vec = embedding.data[0].embedding
    new_memory["embedding"] = emb_vec
    new_memory["benchmark"] = benchmark
    new_memory["task_name"] = task_name
    new_memory["commands"] = commands
    new_memory["judgement"] = judgement
    new_memory["task"] = task
    new_memory["type"] = "summary"
    try:
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            all_memory.append(new_memory)
            new_text_memory = new_memory.copy()
            new_text_memory.pop("embedding")
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)
            # with open(f"{log_dir}/workflow_memory.json", "w") as f:
            #     json.dump({"memory": new_text_memory, "all_memory": [{"task_query": x["task_query"], "workflow": x["workflow"], "benchmark": x["benchmark"], "task_name": x["task_name"], "task": x["task"]} for x in all_memory]}, f, indent=4)
    except Timeout:
        print(
            f"[{benchmark}] Workflow Memory lock timeout "
            f"(>{MEMORY_LOCK_TIMEOUT}s). Skipping workflow memory update for {log_dir}."
        )
        return
    return
    
    
def extract_insight_memory(judgement: bool, trajectory: list, log_dir: str, task_name: str, task: str, commands: list, benchmark: str):
    system_prompt = INSIGHT_CORRECT_PROMPT if judgement else INSIGHT_WRONG_PROMPT
    content = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Trajectory:\n{trajectory}"},
    ]
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=content,
        timeout=300.0,
        service_tier="flex",
    )
    memory_text = response.choices[0].message.content or ""
    memory_path = f"{log_dir.rsplit('/', 2)[0]}/insight_memory.pkl"
    # memory_path = f"{log_dir.rsplit('/', 2)[0]}/memory.pkl"
    lock_path = memory_path + ".lock"
    new_memory = parse_local_memory_items(memory_text)

    embedded_items = []
    # for item in new_memory:
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=new_memory["title"], # + " " + new_memory["description"] + " " + new_memory["content"],
    )
    emb_vec = embedding.data[0].embedding
    embedded_items.append((new_memory, emb_vec))

    # ---------- BEGIN LOCKED SECTION ----------
    try:
        with FileLock(lock_path, timeout=MEMORY_LOCK_TIMEOUT):
            all_memory = []
            # Load existing memory safely
            if os.path.exists(memory_path):
                with open(memory_path, "rb") as f:
                    all_memory = pickle.load(f)
            #     existing_embeddings = [x["embedding"] for x in all_memory]
            # else:
            #     existing_embeddings = []
            # Process and deduplicate memory additions
            appended_memory = []
            for item, emb_vec in embedded_items:
                # duplicate_found = any(
                #     cosine_similarity(emb_vec, e) > 0.7 for e in existing_embeddings
                # )
                # if not duplicate_found:
                all_memory.append({ 
                    "key_embedding": emb_vec,
                    "benchmark": benchmark,
                    "type": "insight",
                    "llm_judge": judgement,
                    "task_name": task_name,
                    "task": task,
                    "insight": new_memory
                })
                appended_memory.append(item)
                # existing_embeddings.append(emb_vec)
            
            with open(memory_path, "wb") as f:
                pickle.dump(all_memory, f)

            # with open(f"{log_dir}/insight_memory.json", "w") as f:
            #     json.dump({"memory": appended_memory, "all_memory": [{"title": x["title"], "description": x["description"], "content": x["content"], "benchmark": x["benchmark"], "task_name": x["task_name"]} for x in all_memory if x["content"] not in [y["content"] for y in appended_memory]]}, f, indent=4)
        # ---------- END LOCKED SECTION ----------
    except Timeout:
        print(
            f"[{benchmark}] Memory lock timeout "
            f"(>{MEMORY_LOCK_TIMEOUT}s). Skipping memory update for {log_dir}."
        )
        return
    return