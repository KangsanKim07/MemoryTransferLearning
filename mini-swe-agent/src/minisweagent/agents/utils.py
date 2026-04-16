from openai import OpenAI
from rank_bm25 import BM25Okapi
import os
import numpy as np
import json
import pickle

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RERANK_PROMPT = """You are a precision reranker for memory retrieval in code/implementation tasks.

Input:
- A Task to solve.
- A list of Memory Items (preselected by embedding similarity, but possibly irrelevant).

Goal:
Select up to THREE memory items that will genuinely help solve the Task.
Do NOT optimize for similarity or surface relevance.
Optimize for real problem-solving usefulness.

Principle:
A memory is useful only if it would realistically be used while solving the Task.
If a memory cannot be directly applied to solving the Task, it is noise and must NOT be selected.
Be conservative in judging whether a memory is actually helpful. When rewriting it, avoid definitive or confident claims, and instead describe it as a potential or optional approach.

Method:
1) Understand what the Task actually requires.
2) Think how you would solve it.
3) Select only memories that would meaningfully influence your decisions, steps, or implementation.
4) Rewrite each memory item as like a guide to solve the given task by making it more applicapable and optimized for the Task. 

Output:
Return only JSON:
[
    {
        "memory_index": int,
        "memory": str
    },
    {
        "memory_index": int,
        "memory": str
    },
    {
        "memory_index": int,
        "memory": str
    }
]

Constraints:
- Max 3 indices, fewer if needed.
- No explanations, no extra text.
"""

def get_embeddings(text: str, chunk_size: int = 10000):
    # words = text.split()
    # chunks = []
    # for i in range(0, len(words), chunk_size):
    #     chunks.append(" ".join(words[i:i+chunk_size]))
    
    resp = client.embeddings.create(model="text-embedding-3-small", input=text[:32000])
    return [resp.data[0].embedding]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_insight_memory_old(benchmark_name: str, plan: str, task: str) -> str:
    
    query_embed = get_embeddings(plan)
    # query_embed = get_embeddings(task)
    memory_path = "/current/" + benchmark_name + "/memory.pkl"
    # # memory_path = "/current/memory.pkl"
    if not os.path.exists(memory_path):
        print("@@@@@@@@@@@ MEMORY PATH NOT FOUND @@@@@@@@@@@", memory_path)
        return ""
    with open(memory_path, "rb") as f:
        memory = pickle.load(f)
    
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    k = 2
    candidates = [x['embedding'] for x in memory]
    scored_memories = []
    for idx, cand in enumerate(candidates):
        max_sim = max(cosine_similarity(q, cand) for q in query_embed)
        scored_memories.append((max_sim, memory[idx]))
    top_memories = sorted(scored_memories, key=lambda x: -x[0])[:k]

    return "\n\nBelow are some memory items that I accumulated from past interaction from the environment that may be helpful to solve the task. You can use it when you feel it’s relevant.\n\n \
In each step, please first explicitly discuss if you want to use each memory item or not, and then take action.\n\n" + "\n\n".join(
        f"# Memory Item {i+2}\n## Title: {m['title']}\n## Description: {m['description']}\n## Content: {m['content']}"
        for i, (_, m) in enumerate(top_memories)
    )


def get_summary_memory(benchmark_name: str, plan: str, task: str) -> str:
    # ===== BM25-based retrieval (currently active) =====
    memory_path = f"/current/{benchmark_name}/summary_memory.pkl"
    if not os.path.exists(memory_path):
        print("@@@@@@@@@@@ MEMORY PATH NOT FOUND @@@@@@@@@@@", memory_path)
        return ""
    with open(memory_path, "rb") as f:
        memory = pickle.load(f)
    
    # def tokenize(text: str) -> list[str]:
    #     return text.lower().split()
    
    # corpus = []
    # for m in memory:
    #     corpus.append(tokenize(m['task']))
    
    # bm25 = BM25Okapi(corpus)
    # query_tokens = tokenize(task)
    # bm25_scores = bm25.get_scores(query_tokens)
    
    # ===== OpenAI Embedding-based retrieval (commented out, uncomment to use) =====   
    query_embed = get_embeddings(task)[0]

    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    task_embeddings = [x['embedding'] for x in memory]
    summary_embeddings = [x['key_embedding'] for x in memory]
    task_scores = []
    for idx, cand in enumerate(task_embeddings):
        task_scores.append(cosine_similarity(query_embed, cand))

    query_goal = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Read the given code editing task and generate a summary containing the goal and required actions in 2-3 sentences."},
            {"role": "user", "content": f"Task: {task}"},
        ],
        timeout=300.0,
    )
    query_goal = query_goal.choices[0].message.content
    query_goal_embedding = get_embeddings(query_goal)[0]
    summary_scores = []
    for cand in summary_embeddings:
        summary_scores.append(cosine_similarity(query_goal_embedding, cand))
    
    # scored_memories = []
    # for i, (bm25_score, task_score, summary_score) in enumerate(zip(bm25_scores, task_scores, summary_scores)):
    #     scored_memories.append((0.33*bm25_score + 0.33*task_score + 0.34*summary_score, memory[i]))
    scored_memories = []
    for i, summary_score in enumerate(summary_scores):
        scored_memories.append((summary_score, memory[i]))

    k = 3
    top_memories = sorted(scored_memories, key=lambda x: -x[0])[:k]
    final_memories = [x for i, x in enumerate(top_memories)]

    print("<retrieval_query>")
    print(query_goal)
    print("</retrieval_query>")
    print("<memory_items>")
    for _, m in top_memories:
        del m['embedding']
        del m['key_embedding']
    print(json.dumps([{"score": sim, "memory": m} for sim, m in top_memories], indent=4))
    print("</memory_items>")

    final_prompt = ""
    for i, (_, m) in enumerate(final_memories):
        result = "PASS" if m['llm_judge'] else "FAIL"
        if result == "PASS":
            final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides a summary extracted from a previous **successful** attempt to solve a similar task.\n\n## Task: {m['summary']['task_summary']}\n\n## Summary: {m['summary']['experience_summary']}\n\n## Result: {result}"
        else:
            final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides a summary extracted from a previous **failed** attempt to solve a similar task.\n\n## Task: {m['summary']['task_summary']}\n\n## Summary: {m['summary']['experience_summary']}\n\n## Result: {result}"

    return final_prompt


def get_insight_memory(benchmark_name: str, plan: str, task: str) -> str:
    # ===== BM25-based retrieval (currently active) =====
    memory_path = f"/current/{benchmark_name}/insight_memory.pkl"
    if not os.path.exists(memory_path):
        print("@@@@@@@@@@@ MEMORY PATH NOT FOUND @@@@@@@@@@@", memory_path)
        return ""
    with open(memory_path, "rb") as f:
        memory = pickle.load(f)

    # ===== OpenAI Embedding-based retrieval (commented out, uncomment to use) =====   
    plan = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Read and think through the task in detail and write a clear, high-level plan for how you will solve it from start to finish. Your plan should be in 3-4 sentences. Do not solve the task, just write the plan."},
            {"role": "user", "content": f"Task: {task}"},
        ],
        timeout=300.0,
    )
    plan = plan.choices[0].message.content
    query_embed = get_embeddings(plan)[0]

    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # task_embeddings = [x['embedding'] for x in memory]
    insight_embeddings = [x['key_embedding'] for x in memory]
    insight_scores = []
    for idx, cand in enumerate(insight_embeddings):
        insight_scores.append(cosine_similarity(query_embed, cand))
    
    scored_memories = []
    for i, insight_score in enumerate(insight_scores):
        scored_memories.append((insight_score, memory[i]))

    k = 3
    top_memories = sorted(scored_memories, key=lambda x: -x[0])[:k]
    final_memories = [x for i, x in enumerate(top_memories)]

    print("<retrieval_query>")
    print(plan)
    print("</retrieval_query>")
    print("<memory_items>")
    for _, m in top_memories:
        if 'embedding' in m:
            del m['embedding']
        if 'key_embedding' in m:
            del m['key_embedding']
    print(json.dumps([{"score": sim, "memory": m} for sim, m in top_memories], indent=4))
    print("</memory_items>")

    if len(final_memories) == 0:
        return ""
    
    final_prompt = "\n\nBelow are insights that I accumulated from past interactions with the environment that may be helpful in solving the task. You can use them when relevant.\n\n\
At each step, please first explicitly state which memory item you want to use."
    for i, (_, m) in enumerate(final_memories):
        result = "PASS" if m['llm_judge'] else "FAIL"
        if result == "PASS":
            final_prompt += "\n\n" + f"# Memory Item {i+1}\n## Title: {m['insight']['title']}\n## Description: {m['insight']['description']}\n## Content: {m['insight']['content']}"
        else:
            final_prompt += "\n\n" + f"# Memory Item {i+1}\n## Title: {m['insight']['title']}\n## Description: {m['insight']['description']}\n## Content: {m['insight']['content']}"

    return final_prompt


def get_traj_memory(benchmark_name: str, plan: str, task: str) -> str:
    # ===== BM25-based retrieval (currently active) =====
    # memory_path = "/current/memory.pkl"
    memory_path = f"/current/{benchmark_name}/traj_memory.pkl"
    try:
        with open(memory_path, "rb") as f:
            memory = pickle.load(f)
    except Exception as e:
        print("@@@@@@@@@@@@@ MEMORY PATH NOT FOUND @@@@@@@@@@@@@@", memory_path)
        return ""
    
    # def tokenize(text: str) -> list[str]:
    #     return text.lower().split()
    
    # corpus = []
    # for m in memory:
    #     corpus.append(tokenize(m['task']))
    
    # bm25 = BM25Okapi(corpus)
    # query_tokens = tokenize(task)
    # scores = bm25.get_scores(query_tokens)
    
    # first_k = min(30, int(len(memory)*0.1))
    # top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:first_k]
    # memory = [memory[i] for i in top_indices]

    # ===== OpenAI Embedding-based retrieval (commented out, uncomment to use) =====   
    # query_embed = get_embeddings(plan)
    query_embed = get_embeddings(task)

    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # local_memory_path = "/current/" + benchmark_name + "/trajectory_memory.pkl"
    # local_memory_path = "/current/memory.pkl"
    # if not os.path.exists(memory_path):
    #     print("@@@@@@@@@@@ TRAJECTORY MEMORY PATH NOT FOUND @@@@@@@@@@@", memory_path)
    #     return ""
    # with open(memory_path, "rb") as f:
    #     local_memory = pickle.load(f)
    local_candidates = [x['embedding'] for x in memory]#local_memory]
    local_scored_memories = []
    for idx, cand in enumerate(local_candidates):
        # max_sim = max(0.5*cosine_similarity(q, cand) + 0.5*scores[idx] for q in query_embed)
        sim = cosine_similarity(query_embed[0], cand)
        # if local_memory[idx]['judgement']:
        local_scored_memories.append((sim, memory[idx]))
    k = 3
    local_top_memories = sorted(local_scored_memories, key=lambda x: -x[0])[:k]
    for m in local_top_memories:
        del m[1]['embedding']

    print("<memory_items>")
    print(json.dumps([{"score": sim, "memory": m} for sim, m in local_top_memories], indent=4))
    print("</memory_items>")

#     memory_items = ""
#     for i, (_, m) in enumerate(local_top_memories):
#         result = "PASS" if m['judgement'] else "FAIL"
#         if "workflow" in m:
#             memory_items += f"# Memory Item {i}\n## Goal: {m['goal']}\n## Workflow: {m['workflow']}\n## Result: {result}\n\n"
#         elif "task_summary" in m:
#             memory_items += f"# Memory Item {i}\n## Task: {m['task_summary']}\n## Experience: {m['experience_summary']}\n## Result: {result}\n\n"
#         elif "title" in m:
#             memory_items += f"# Memory Item {i}\n## Title: {m['title']}\n## Description: {m['description']}\n## Content: {m['content']}\n## Result: {result}\n\n"
#         else:
#             raise ValueError("Unknown memory type")

#     rerank_result = client.chat.completions.create(
#         model="gpt-5-mini",
#         messages=[
#             {"role": "system", "content": """You are a helpful reranker for code editing memory item retrieval. You will be given a task and a list of memory items that are relevant to the task. You need to select at most three memory items which you think are greatly helpful to solve the task.\
# Answer in a JSON dict with a 'memory_index' key with a list of three memory indices. Select no more than three memory items. If fewer than three relevant memories exist, select fewer than three.\nAnswer Format: {"memory_index": [0, 2, 4]}"""},
#             {"role": "user", "content": f"Task: {task}\n\n{memory_items}"},
#         ],
#         timeout=300.0,
#     )

#     rerank_response = rerank_result.choices[0].message.content
#     rerank_response = json.loads(rerank_response)
#     rerank_indices = rerank_response['memory_index']
#     if len(rerank_indices) == 0:
#         return ""
    final_memories = [x for i, x in enumerate(local_top_memories)]# if i in rerank_indices]
    for _, m in final_memories:
        for t in m['trajectory']:
            if t['role'] == 'observation':
                t['message'] = t['message'][:300]

    final_prompt = "\n\nBelow are trajectories that I accumulated from past interactions with the environment that may be helpful in solving the task. You can use them when relevant.\n\n\
At each step, please first explicitly state which memory item you want to use. Since LLM judgments are not always correct, you need to verify the memory items before using them and do NOT blindly follow failed attempts."

    for i, (_, m) in enumerate(final_memories):
        result = "PASS" if m['llm_judge'] else "FAIL"
        if result == "PASS":
            final_prompt += "\n\n" + f"# Memory Item {i+1}\n\n## Task: {m['task']}\n\n## Trajectory: {json.dumps(m['trajectory'], indent=2)}\n\n## LLM Judgment: {result}"
        else:
            final_prompt += "\n\n" + f"# Memory Item {i+1}\n\n## Task: {m['task']}\n\n## Trajectory: {json.dumps(m['trajectory'], indent=2)}\n\n## LLM Judgment: {result}"

    return final_prompt


def get_workflow_memory(benchmark_name: str, plan: str, task: str) -> str:
    # ===== BM25-based retrieval (currently active) =====
    memory_path = f"/current/{benchmark_name}/workflow_memory.pkl"
    if not os.path.exists(memory_path):
        print("@@@@@@@@@@@ MEMORY PATH NOT FOUND @@@@@@@@@@@", memory_path)
        return ""
    with open(memory_path, "rb") as f:
        memory = pickle.load(f)
    
    # def tokenize(text: str) -> list[str]:
    #     return text.lower().split()
    
    # corpus = []
    # for m in memory:
    #     corpus.append(tokenize(m['task']))
    
    # bm25 = BM25Okapi(corpus)
    # query_tokens = tokenize(task)
    # bm25_scores = bm25.get_scores(query_tokens)
    
    # ===== OpenAI Embedding-based retrieval (commented out, uncomment to use) =====   
    # query_embed = get_embeddings(task)[0]

    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # task_embeddings = [x['embedding'] for x in memory]
    goal_embeddings = [x['key_embedding'] for x in memory]
    # task_scores = []
    # for idx, cand in enumerate(task_embeddings):
    #     task_scores.append(cosine_similarity(query_embed, cand))

    query_goal = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Read and think through the task in detail and write a clear, high-level plan for how you will solve it from start to finish. Your plan should be in 3-4 sentences. Do not solve the task, just write the plan."},
            {"role": "user", "content": f"Task: {task}"},
        ],
        timeout=300.0,
    )
    query_goal = query_goal.choices[0].message.content
    query_goal_embedding = get_embeddings(query_goal)[0]
    goal_scores = []
    for cand in goal_embeddings:
        goal_scores.append(cosine_similarity(query_goal_embedding, cand))
    
    scored_memories = []
    for i, goal_score in enumerate(goal_scores):
        scored_memories.append((goal_score, memory[i]))

    k = 3
    top_memories = sorted(scored_memories, key=lambda x: -x[0])[:k]
    final_memories = [x for i, x in enumerate(top_memories)]

    print("<retrieval_query>")
    print(query_goal)
    print("</retrieval_query>")
    print("<memory_items>")
    for _, m in top_memories:
        del m['embedding']
        del m['key_embedding']
    print(json.dumps([{"score": sim, "memory": m} for sim, m in top_memories], indent=4))
    print("</memory_items>")

    final_prompt = "\n\nBelow are workflows that I accumulated from past interactions with the environment that may be helpful in solving the task. You can use them when relevant.\n\n\
At each step, please first explicitly state which memory item you want to use. Since LLM judgments are not always correct, you need to verify the memory items before using them and do NOT blindly follow failed attempts."
    for i, (_, m) in enumerate(final_memories):
        result = "PASS" if m['llm_judge'] else "FAIL"
        if result == "PASS":
            final_prompt += "\n\n" + f"# Memory Item {i+1}\n\n## Goal: {m['workflow']['goal']}\n\n## Workflow: {m['workflow']['workflow']}\n\n"
        else:
            final_prompt += "\n\n" + f"# Memory Item {i+1}\n\n## Goal: {m['workflow']['goal']}\n\n## Workflow: {m['workflow']['workflow']}\n\n"## LLM Judge: {result}"

    return final_prompt


def get_workflow_insight_memory(benchmark_name: str, plan: str, task: str) -> str:
    with open(f"/current/{benchmark_name}/workflow_memory.pkl", "rb") as f:
        workflow_memory = pickle.load(f)
        workflow_embeddings = [x['key_embedding'] for x in workflow_memory]
    with open(f"/current/{benchmark_name}/insight_memory.pkl", "rb") as f:
        insight_memory = pickle.load(f)
        insight_embeddings = [x['key_embedding'] for x in insight_memory]
    
    # all_memory = workflow_memory + insight_memory
    query_goal = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Read the given code editing task and generate a summary containing the goal and required actions in 2-3 sentences."},
            {"role": "user", "content": f"Task: {task}"},
        ],
        timeout=300.0,
    )
    query_goal = query_goal.choices[0].message.content
    query_goal_embedding = get_embeddings(query_goal)[0]
    scores = []
    for cand in workflow_embeddings + insight_embeddings:
        scores.append(cosine_similarity(query_goal_embedding, cand))

    scored_memories = []
    total_memory = workflow_memory + insight_memory
    for i, score in enumerate(scores):
        scored_memories.append((score, total_memory[i]))

    k = 3
    top_memories = sorted(scored_memories, key=lambda x: -x[0])[:k]

    print("<memory_items>")
    for _, m in top_memories:
        del m['embedding']
        del m['key_embedding']
    print(json.dumps([{"score": sim, "memory": m} for sim, m in top_memories], indent=4))
    print("</memory_items>")

    final_prompt = ""
    for i, (_, m) in enumerate(top_memories):
        result = "PASS" if m['llm_judge'] else "FAIL"
        if result == "PASS":
            if 'workflow' in m:
                final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides a workflow extracted from a previous attempt that LLM judged to be successful at solving a similar task. Be aware that LLM judgments are not always correct.\n\n## Goal: {m['workflow']['goal']}\n\n## Workflow: {m['workflow']['workflow']}\n\n## LLM Judge: {result}"
            else:
                final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides an insight extracted from a previous attempt that LLM judged to be successful to solve a similar task. Be aware that LLM judgments are not always correct.\n\n## Title: {m['insight']['title']}\n\n## Description: {m['insight']['description']}\n\n## Content: {m['insight']['content']}\n\n## LLM Judge: {result}"
        else:
            if 'workflow' in m:
                final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides a failed workflow extracted from a previous attempt that LLM judged to be unsuccessful at solving a similar task. Be aware that LLM judgments are not always correct.\n\n## Goal: {m['workflow']['goal']}\n\n## Workflow: {m['workflow']['workflow']}\n\n## LLM Judge: {result}"
            else:
                final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides an insight extracted from a previous attempt that LLM judged to be unsuccessful to solve a similar task. Be aware that LLM judgments are not always correct.\n\n## Title: {m['insight']['title']}\n\n## Description: {m['insight']['description']}\n\n## Content: {m['insight']['content']}\n\n## LLM Judge: {result}"

    return final_prompt


def get_workflow_insight_memory_rerank(benchmark_name: str, plan: str, task: str) -> str:
    with open(f"/current/{benchmark_name}/workflow_memory.pkl", "rb") as f:
        workflow_memory = pickle.load(f)
        workflow_embeddings = [x['key_embedding'] for x in workflow_memory]
    with open(f"/current/{benchmark_name}/insight_memory.pkl", "rb") as f:
        insight_memory = pickle.load(f)
        insight_embeddings = [x['key_embedding'] for x in insight_memory]
    
    # all_memory = workflow_memory + insight_memory
    query_goal = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Read the given code editing task and generate a summary containing the goal and required actions in 2-3 sentences."},
            {"role": "user", "content": f"Task: {task}"},
        ],
        timeout=300.0,
    )
    query_goal = query_goal.choices[0].message.content
    query_goal_embedding = get_embeddings(query_goal)[0]
    workflow_scores = []
    for cand in workflow_embeddings:
        workflow_scores.append(cosine_similarity(query_goal_embedding, cand))
    insight_scores = []
    for cand in insight_embeddings:
        insight_scores.append(cosine_similarity(query_goal_embedding, cand))

    workflow_scored_memories = []
    for i, workflow_score in enumerate(workflow_scores):
        workflow_scored_memories.append((workflow_score, workflow_memory[i]))
    insight_scored_memories = []
    for i, insight_score in enumerate(insight_scores):
        insight_scored_memories.append((insight_score, insight_memory[i]))

    k = 10
    workflow_top_memories = sorted(workflow_scored_memories, key=lambda x: -x[0])[:k]
    insight_top_memories = sorted(insight_scored_memories, key=lambda x: -x[0])[:k]
    
    
    memory_items = ""
    for i, (_, m) in enumerate(workflow_top_memories + insight_top_memories):
        result = "PASS" if m['llm_judge'] else "FAIL"
        if "workflow" in m:
            memory_items += f"# Memory Item {i}\n## Goal: {m['workflow']['goal']}\n## Workflow: {m['workflow']['workflow']}\n## LLM Judge: {result}\n\n"
        elif "insight" in m:
            memory_items += f"# Memory Item {i}\n## Title: {m['insight']['title']}\n## Description: {m['insight']['description']}\n## Content: {m['insight']['content']}\n## LLM Judge: {result}\n\n"
        else:
            raise ValueError("Unknown memory type")
    
    rerank_result = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": RERANK_PROMPT},
            {"role": "user", "content": f"Task: {task}\n\n{memory_items}"},
        ],
        timeout=300.0,
    )

    # rerank_response = rerank_result.choices[0].message.content
    # rerank_response = json.loads(rerank_response)
    # rerank_indices = rerank_response['memory_index']
    # if len(rerank_indices) == 0:
    #     return ""
    # final_memories = [x for i, x in enumerate(workflow_top_memories + insight_top_memories) if i in rerank_indices]

    rerank_response = rerank_result.choices[0].message.content
    rerank_response = json.loads(rerank_response)
    print("<rerank_response>")
    print(rerank_response)
    print("</rerank_response>")
    rerank_dict = {d['memory_index']: d['memory'] for d in rerank_response}
    final_memories = []
    for i, (s, m) in enumerate(workflow_top_memories + insight_top_memories):
        if i in rerank_dict:
            final_memories.append((s, {**m, 'rewrite': rerank_dict[i]}))

    print("<retrieval_query>")
    print(query_goal)
    print("</retrieval_query>")
    print("<memory_items>")
    for _, m in final_memories:
        del m['embedding']
        del m['key_embedding']
    print(json.dumps([{"score": sim, "memory": m} for sim, m in final_memories], indent=4))
    print("</memory_items>")

    final_prompt = ""
    for i, (_, m) in enumerate(final_memories):
        result = "PASS" if m['llm_judge'] else "FAIL"
        # if result == "PASS":
        #     if 'workflow' in m:
        #         final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides a workflow extracted from a previous attempt that LLM judged to be successful at solving a similar task. Be aware that LLM judgments are not always correct.\n\n## Goal: {m['workflow']['goal']}\n\n## Workflow: {m['workflow']['workflow']}\n\n## LLM Judge: {result}"
        #     else:
        #         final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides an insight extracted from a previous attempt that LLM judged to be successful to solve a similar task. Be aware that LLM judgments are not always correct.\n\n## Title: {m['insight']['title']}\n\n## Description: {m['insight']['description']}\n\n## Content: {m['insight']['content']}\n\n## LLM Judge: {result}"
        # else:
        #     if 'workflow' in m:
        #         final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides a failed workflow extracted from a previous attempt that LLM judged to be unsuccessful at solving a similar task. Be aware that LLM judgments are not always correct.\n\n## Goal: {m['workflow']['goal']}\n\n## Workflow: {m['workflow']['workflow']}\n\n## LLM Judge: {result}"
        #     else:
        #         final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides an insight extracted from a previous attempt that LLM judged to be unsuccessful to solve a similar task. Be aware that LLM judgments are not always correct.\n\n## Title: {m['insight']['title']}\n\n## Description: {m['insight']['description']}\n\n## Content: {m['insight']['content']}\n\n## LLM Judge: {result}"
        final_prompt += "\n\n" + f"# Memory Item {i+1}\n\n## Memory: {m['rewrite']}"

    return final_prompt


def get_good_insights():
    with open("/current/good_memories_filtered.json", "r") as f:
        good_memories = json.load(f)
    prompt = ""
    for i, m in enumerate(good_memories):
        prompt += f"[Guideline {i+1}]\nTitle: {m['insight']['title']}\nDescription: {m['insight']['description']}\nContent: {m['insight']['content']}\n\n"
    return prompt


def get_wi_memory(benchmark_name: str, plan: str, task: str) -> str:
    if os.path.exists(f"/current/{benchmark_name}/workflow_memory.pkl"):
        with open(f"/current/{benchmark_name}/workflow_memory.pkl", "rb") as f:
            workflow_memory = pickle.load(f)
            workflow_embeddings = [x['key_embedding'] for x in workflow_memory]
    else:
        workflow_memory = []
        workflow_embeddings = []
    if os.path.exists(f"/current/{benchmark_name}/insight_memory.pkl"):
        with open(f"/current/{benchmark_name}/insight_memory.pkl", "rb") as f:
            insight_memory = pickle.load(f)
            insight_embeddings = [x['key_embedding'] for x in insight_memory]
    else:
        insight_memory = []
        insight_embeddings = []
    if workflow_memory == [] and insight_memory == []:
        return ""
    
    plan = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful planner for a code generation/editing agent. Read and think through the given task in detail and write a clear, high-level plan for how you will solve it from start to finish. Your plan should be in 4-5 sentences."},
            {"role": "user", "content": f"Task: {task}"},
        ],
        timeout=300.0,
    )
    plan = plan.choices[0].message.content
    query_embedding = get_embeddings(plan)[0]
    scores = []
    for cand in workflow_embeddings + insight_embeddings:
        scores.append(cosine_similarity(query_embedding, cand))
    util_scores = []
    for cand in workflow_memory + insight_memory:
        util_scores.append(cand['util_score'])

    scored_memories = []
    total_memory = workflow_memory + insight_memory
    for i, score in enumerate(scores):
        scored_memories.append((score, total_memory[i]))

    top_memories = sorted(scored_memories, key=lambda x: -x[0])[:20]
    top_memories = sorted(top_memories, key=lambda x: -x[1]['util_score'])[:3]

    print("<plan>")
    print(plan)
    print("</plan>")
    print("<memory_items>")
    for _, m in top_memories:
        if 'embedding' in m:
            del m['embedding']
        if 'key_embedding' in m:
            del m['key_embedding']
    print(json.dumps([{"score": sim, "memory": m} for sim, m in top_memories], indent=4))
    print("</memory_items>")

    final_prompt = "These are memories that I accumulated from past code editing attempts that may be helpful to solve the given task. You can apply some of them to solve the given task, and ignore the unrelated memories among them.\
In each step, please first explicitly discuss which memory item you want to use."
# If you want to use the memories, please explicitly discuss which memory item you want to use at the beginning of the step."
    retrieved_workflows = [x for _, x in top_memories if 'workflow' in x]
    retrieved_insights = [x for _, x in top_memories if 'insight' in x]
    memory_count = 0
    workflow_prompt = ""
    for m in retrieved_workflows:
        workflow_prompt += "\n\n" + f"# Memory Item {memory_count+1}\n## Goal: {m['workflow']['goal']}\n## Workflow: {m['workflow']['workflow']}"
        memory_count += 1
    if workflow_prompt != "":
        final_prompt = final_prompt + "\n\n" + "These memories provide workflows extracted from previous attempts to solve different tasks within the same benchmark." + workflow_prompt
    
    insight_prompt = ""
    for m in retrieved_insights:
        insight_prompt += "\n\n" + f"# Memory Item {memory_count+1}\n## Title: {m['insight']['title']}\n## Description: {m['insight']['description']}\n## Content: {m['insight']['content']}"
        memory_count += 1
    if insight_prompt != "":
        final_prompt = final_prompt + "\n\n" + "These memories provide insights extracted from previous attempts to solve different tasks across different benchmarks." + insight_prompt

    return final_prompt

def get_insight_memory_rerank(benchmark_name: str, plan: str, task: str) -> str:
    with open(f"/current/{benchmark_name}/insight_memory.pkl", "rb") as f:
        insight_memory = pickle.load(f)
        insight_embeddings = [x['key_embedding'] for x in insight_memory]
    
    # all_memory = workflow_memory + insight_memory
    query_goal = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Read the given code editing task and generate a summary containing the goal and required actions in 2-3 sentences."},
            {"role": "user", "content": f"Task: {task}"},
        ],
        timeout=300.0,
    )
    query_goal = query_goal.choices[0].message.content
    query_goal_embedding = get_embeddings(query_goal)[0]
    insight_scores = []
    for cand in insight_embeddings:
        insight_scores.append(cosine_similarity(query_goal_embedding, cand))

    insight_scored_memories = []
    for i, insight_score in enumerate(insight_scores):
        insight_scored_memories.append((insight_score, insight_memory[i]))

    k = 20
    insight_top_memories = sorted(insight_scored_memories, key=lambda x: -x[0])[:k]
    
    
    memory_items = ""
    for i, (_, m) in enumerate(workflow_top_memories + insight_top_memories):
        result = "PASS" if m['llm_judge'] else "FAIL"
        if "workflow" in m:
            memory_items += f"# Memory Item {i}\n## Goal: {m['workflow']['goal']}\n## Workflow: {m['workflow']['workflow']}\n## LLM Judge: {result}\n\n"
        elif "insight" in m:
            memory_items += f"# Memory Item {i}\n## Title: {m['insight']['title']}\n## Description: {m['insight']['description']}\n## Content: {m['insight']['content']}\n## LLM Judge: {result}\n\n"
        else:
            raise ValueError("Unknown memory type")
    
    rerank_result = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": RERANK_PROMPT},
            {"role": "user", "content": f"Task: {task}\n\n{memory_items}"},
        ],
        timeout=300.0,
    )

    # rerank_response = rerank_result.choices[0].message.content
    # rerank_response = json.loads(rerank_response)
    # rerank_indices = rerank_response['memory_index']
    # if len(rerank_indices) == 0:
    #     return ""
    # final_memories = [x for i, x in enumerate(workflow_top_memories + insight_top_memories) if i in rerank_indices]

    rerank_response = rerank_result.choices[0].message.content
    rerank_response = json.loads(rerank_response)
    print("<rerank_response>")
    print(rerank_response)
    print("</rerank_response>")
    rerank_dict = {d['memory_index']: d['memory'] for d in rerank_response}
    final_memories = []
    for i, (s, m) in enumerate(workflow_top_memories + insight_top_memories):
        if i in rerank_dict:
            final_memories.append((s, {**m, 'rewrite': rerank_dict[i]}))

    print("<retrieval_query>")
    print(query_goal)
    print("</retrieval_query>")
    print("<memory_items>")
    for _, m in final_memories:
        del m['embedding']
        del m['key_embedding']
    print(json.dumps([{"score": sim, "memory": m} for sim, m in final_memories], indent=4))
    print("</memory_items>")

    final_prompt = ""
    for i, (_, m) in enumerate(final_memories):
        result = "PASS" if m['llm_judge'] else "FAIL"
        # if result == "PASS":
        #     if 'workflow' in m:
        #         final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides a workflow extracted from a previous attempt that LLM judged to be successful at solving a similar task. Be aware that LLM judgments are not always correct.\n\n## Goal: {m['workflow']['goal']}\n\n## Workflow: {m['workflow']['workflow']}\n\n## LLM Judge: {result}"
        #     else:
        #         final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides an insight extracted from a previous attempt that LLM judged to be successful to solve a similar task. Be aware that LLM judgments are not always correct.\n\n## Title: {m['insight']['title']}\n\n## Description: {m['insight']['description']}\n\n## Content: {m['insight']['content']}\n\n## LLM Judge: {result}"
        # else:
        #     if 'workflow' in m:
        #         final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides a failed workflow extracted from a previous attempt that LLM judged to be unsuccessful at solving a similar task. Be aware that LLM judgments are not always correct.\n\n## Goal: {m['workflow']['goal']}\n\n## Workflow: {m['workflow']['workflow']}\n\n## LLM Judge: {result}"
        #     else:
        #         final_prompt += "\n\n" + f"# Memory Item {i+1}\nThis memory provides an insight extracted from a previous attempt that LLM judged to be unsuccessful to solve a similar task. Be aware that LLM judgments are not always correct.\n\n## Title: {m['insight']['title']}\n\n## Description: {m['insight']['description']}\n\n## Content: {m['insight']['content']}\n\n## LLM Judge: {result}"
        final_prompt += "\n\n" + f"# Memory Item {i+1}\n\n## Memory: {m['rewrite']}"

    return final_prompt