CODE_CORRECT_PROMPT = """You are an expert in repository-level code editing. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the task.
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first think why the trajectory is successful, and then summarize the insights.
- You can extract at most 2 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific files, queries, or string contents, but rather focus on the generalizable insights.
## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```
"""

CODE_WRONG_PROMPT = """You are an expert in repository-level code editing. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the task but failed.
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
- You can extract at most 2 memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific files, queries, or string contents, but rather focus on the generalizable insights.
## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```
"""

CODE_CORRECT_ONE_MEMORY_PROMPT = """You are an expert in repository-level code editing. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the task.
## Guidelines
You need to extract and summarize one useful insight in the format of memory item based on the agent's successful trajectory.
The goal of summarized memory item is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first think why the trajectory is successful, and then summarize the insights.
- You must not repeat similar or overlapping items with the previous memory items in the system prompt.
## Output Format
Your output must strictly follow the Markdown format shown below:
```
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task> 
```
"""

CODE_WRONG_ONE_MEMORY_PROMPT = """You are an expert in repository-level code editing. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the task but failed.
## Guidelines
You need to extract and summarize one useful insight in the format of memory item based on the agent's failed trajectory.
The goal of summarized memory item is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
- You must not repeat similar or overlapping items with the previous memory items in the system prompt.
## Output Format
Your output must strictly follow the Markdown format shown below:
```
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task> 
```
"""

# - Do not mention specific files, queries, or string contents, but rather focus on the generalizable insights.
# - Please summarize the setting, problem, your trajectory, solution, and final insights in the memory item. But they should be not too long or too detailed.
# <1-3 sentences describing the summary of the setting, problem, your trajectory, solution, and final insights> 


CODE_SPECIFIC_CORRECT_PROMPT = """You are an expert in repository-level code editing. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the task.
## Guidelines
You need to summarize useful reasoning patterns in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first think why the trajectory is successful, and then summarize the reasoning patterns to reuse in future similar tasks.
- You can extract only one memory item from the trajectory.
- This memory item will be used for solving similar tasks in the same benchmark.
### Output Format
Your output must strictly follow the json object format shown below:
```json
{
    "when_to_use" : "Specific scenarios where this reasoning pattern applies", 
    "task_query" : "Summarize the task query given by the user.", 
    "generalized_query" : "Abstract the specific task query to create a more generalized task representation.", 
    "experience" : "Detailed analysis of what made the this trajectory successful to accomplish the task and how to reuse the reasoning pattern in future similar tasks.", 
    "tags" : ["relevant", "keywords", "from", "the", "task", "query"]
}
```
"""

CODE_SPECIFIC_WRONG_PROMPT = """You are an expert in repository-level code editing. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the task but failed.
## Guidelines
You need to summarize useful reasoning patterns in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first think why the trajectory failed, and then summarize the reasoning patterns to prevent the failure in future similar tasks.
- You can extract only one memory item from the trajectory.
- This memory item will be used for solving similar tasks in the same benchmark.
### Output Format
Your output must strictly follow the json object format shown below:
```json
{
    "when_to_use" : "Specific scenarios where this reasoning pattern applies", 
    "task_query" : "Summarize the task query given by the user.", 
    "generalized_query" : "Abstract the specific task query to create a more generalized task representation.", 
    "experience" : "Detailed analysis of what made the this trajectory failed to accomplish the task and how to prevent the failure in future similar tasks.", 
    "tags" : ["relevant", "keywords", "from", "the", "task", "query"]
}
```
"""

WORKFLOW_CORRECT_EXTRACT_PROMPT = """## Task Description
You are given a **successful command trajectory** for a code-editing task — a sequence of Bash commands that correctly achieved its goal.
Your goal is to extract a **single reusable base workflow** that captures the **core strategy behind the success**, so it can be reused in similar future tasks.
The workflow should abstract the effective problem-solving pattern, such as how files were found, edited, and verified.

## What You Need to Produce
You must identify:
* The **goal** of the workflow
  Describe what kind of subproblem it solves and when it should be used.
* The **sequence of Bash commands**
  Include the key command steps that made the trajectory succeed, focusing on reusable patterns rather than one-off details.

## Workflow Requirements
The workflow must satisfy all of the following:
* It must come from the **given successful trajectory**
* It must be a **reusable subroutine**, not just a command log
* It should reflect **why the trajectory worked**, such as how files were found, code was modified, and results were checked
* Long commands may be **shortened** as long as the core action remains

## Output Format
Your output must follow this exact JSON format:
```json
{
  "goal": "Describe when this workflow can be applied.",
  "workflow": [
    "bash command 1",
    "bash command 2",
    "bash command 3",
    ...
  ]
}
```"""

WORKFLOW_WRONG_EXTRACT_PROMPT = """## Task Description
You are given a **failed command trajectory** for a code-editing task — a sequence of Bash commands that did not reach the intended goal.
Your goal is to extract a **single reusable base workflow** that captures a **bad or misleading strategy** that led to failure, so it can be avoided in similar future tasks.
The workflow should abstract the **anti-pattern** behind the failure, such as skipping inspection, editing the wrong files, or making unchecked changes.

## What You Need to Produce
You must identify:
* The **goal** of the workflow
  Describe what kind of mistake or failure pattern this workflow represents and when it tends to occur.
* The **sequence of Bash commands**
  Include the key command steps that illustrate the **incorrect or harmful pattern**, rather than how it was fixed.

## Workflow Requirements
The workflow must satisfy all of the following:
* It must come from the **given failed trajectory**
* It must be a **reusable negative example**, not just a random error
* It should reflect **why the trajectory failed**, such as skipping checks, editing blindly, or validating the wrong thing
* Long commands may be **shortened** as long as the core action remains

## Output Format
Your output must follow this exact JSON format:
```json
{
  "goal": "Describe when this workflow can be applied.",
  "workflow": [
    "bash command 1",
    "bash command 2",
    "bash command 3",
    ...
  ]
}
```"""

SUMMARY_CORRECT_EXTRACT_PROMPT = """## Task Description
You are given a **successful command trajectory** for a code-editing task - a sequence of actions and Bash commands that correctly achieved its goal.
Your goal is to produce a **structured summary** that captures both what was done and why it worked, so this trajectory can be used as a reference for solving similar tasks in the future.
The summary should not just restate the commands, but explain the **task context, the approach taken, and the factors that led to success**.

## What You Need to Produce
You must produce two summaries:
* **task_summary**
  A short description of the task that was being solved, written in two or three sentences.
  It should describe what needed to be changed or built and in what kind of codebase or environment.
* **experience_summary**
  A one-paragraph summary of the entire trajectory.
  It should describe the code environment, the key actions or commands, the overall approach, and the final outcome.
  It should also explain why this trajectory succeeded, highlighting useful strategies, checks, or decisions that contributed to the correct result.

## Summary Requirements
The summaries must satisfy all of the following:
* They must be based only on the **given successful trajectory**
* They must be written so they are **useful as reference material** for future similar tasks
* The **experience_summary** should capture the **important technical and strategic details**
* The **experience_summary** should include some analysis of **why the approach worked**, not just what happened

## Output Format
Your output must follow this exact JSON format:
```json
{
  "task_summary": "A short description of the task and what needed to be accomplished.",
  "experience_summary": "A one-paragraph explanation of the trajectory, including the environment, key actions, approach, outcome, and why it succeeded."
}
```"""

SUMMARY_WRONG_EXTRACT_PROMPT = """## Task Description
You are given a **failed command trajectory** for a code-editing task — a sequence of actions and Bash commands that did not achieve the intended goal or produced incorrect results.
Your goal is to produce a **structured summary** that captures what was attempted, what went wrong, and why this trajectory is a useful negative example for future tasks.
The summary should not just list mistakes, but explain the **task context, the approach taken, and the reasons the outcome was incorrect or incomplete**.

## What You Need to Produce
You must produce two summaries:
* **task_summary**
  A short description of the task that was being attempted, written in two or three sentences.
  It should describe what needed to be changed or built and in what kind of codebase or environment.
* **experience_summary**
  A one-paragraph summary of the entire trajectory.
  It should describe the given task, the code environment, the key actions or commands, the overall approach, and the final (failed) outcome.
  It should also explain why this trajectory failed, highlighting incorrect assumptions, missing checks, or flawed strategies so that others can avoid repeating them.

## Summary Requirements
The summaries must satisfy all of the following:
* They must be based only on the **given failed trajectory**
* They must be written so they are **useful as reference material** for future similar tasks
* The **experience_summary** should capture the **important technical and strategic details**
* The **experience_summary** should include analysis of **why the approach did not work**, not just what happened

## Output Format
Your output must follow this exact JSON format:
```json
{
  "task_summary": "A short description of the task that was being attempted.",
  "experience_summary": "A one-paragraph explanation of what was done, what and why went wrong, and how to avoid same failure in the future tasks."
}
```"""

INSIGHT_CORRECT_PROMPT = """You are an expert in repository-level code editing. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the task.
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first think why the trajectory is successful, and then summarize the insights.
- You can extract only one memory item from the trajectory.
- Do not mention specific files, queries, or string contents, but rather focus on the generalizable insights.
## Output Format
Your output must strictly follow the json format shown below:
```json
{
  "title": "the title of the memory item",
  "description": "one sentence summary of the memory item",
  "content": "1-3 sentences describing the insights learned to successfully accomplishing the task"
}
```
"""

INSIGHT_WRONG_PROMPT = """You are an expert in repository-level code editing. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the task but failed.
## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
- You can extract only one memory item from the trajectory.
- Do not mention specific files, queries, or string contents, but rather focus on the generalizable insights.
## Output Format
Your output must strictly follow the json format shown below:
```json
{
  "title": "the title of the memory item",
  "description": "one sentence summary of the memory item",
  "content": "1-3 sentences describing the insights learned to successfully accomplishing the task"
}
```
"""