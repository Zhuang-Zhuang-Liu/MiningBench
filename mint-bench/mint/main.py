from mint.envs import GeneralEnv, AlfworldEnv
from mint.datatypes import Action, State
from mint.tasks import AlfWorldTask
from mint.tools import Tool
import mint.tasks as tasks
import mint.agents as agents
import logging
import os
import json
import pathlib
import importlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Configure logging settings
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("MINT")


def interactive_loop(
    task: tasks.Task,
    agent: agents.LMAgent,
    tools: List[Tool],
    feedback_config: Dict[str, Any],
    env_config: Dict[str, Any],
    interactive_mode: bool = False,
    worker_id: int = 0,
):
    if AlfWorldTask is not None and isinstance(task, AlfWorldTask):
        LOGGER.info(f"[Worker {worker_id}] loading Alfworld Env")
        env = AlfworldEnv(task, tools, feedback_config, env_config)
    else:
        env = GeneralEnv(task, tools, feedback_config, env_config)
    state: State = env.reset()

    init_msg = state.latest_output['content']
    if interactive_mode:
        # omit in-context example
        splited_msg = init_msg.split("---")
        init_msg = splited_msg[0] + "== In-context Example Omitted ==" + splited_msg[2]

    LOGGER.info(f"\n[Worker {worker_id}] User: \n\033[94m{state.latest_output['content']}\033[0m")

    num_steps = 0

    if task.loaded_history is not None:
        for turn in task.loaded_history:
            action = agent.lm_output_to_action(turn["lm_output"])
            LOGGER.info(
                f"\n[Worker {worker_id}] Loaded LM Agent Action:\n\033[92m{action.value}\033[0m")
            state = env.step(action, loaded=turn)
            LOGGER.info(
                "\033[1m" + f"[Worker {worker_id}] User:\n" + "\033[0m" +
                f"\033[94m{state.latest_output['content']}\033[0m"
            )
            num_steps += 1

    while not state.finished:
        # agent act
        if interactive_mode:
            to_continue = "n"
            while to_continue not in ["y", "Y"]:
                to_continue = input("\n> Continue? (y/n) ")

        action: Action = agent.act(state)
        # color the action in green
        # LOGGER.info(f"\nLM Agent Action:\n\033[92m{action.value}\033[0m")
        LOGGER.info(
            f"\n\033[1m" + f"[Worker {worker_id}] LM Agent Action:\n" + "\033[0m" +
            f"\n\033[92m{action.value}\033[0m"
        )
        # environment step
        state: State = env.step(action)
        # color the state in blue
        if not state.finished:
            user_msg = state.latest_output['content']
            if "Expert feedback:" in user_msg:
                try:
                    splited = user_msg.split("Expert feedback:")
                    obs, feedback = splited[0], "".join(splited[1:])
                except Exception as e:
                    LOGGER.info(f"[Worker {worker_id}] Error: {e}")
                    LOGGER.info(f"[Worker {worker_id}] User message: {user_msg}")
                    LOGGER.exception(e)
                    raise e

                feedback = "Expert feedback:" + feedback
                # color the observation in blue & feedback in red
                LOGGER.info(
                    "\n" +
                    "\033[1m" + f"[Worker {worker_id}] User:\n" + "\033[0m" +
                    f"\033[94m{obs}\033[0m" + "\n" 
                    + f"\033[93m{feedback}\033[0m" + "\n"
                )
            else:
                # color the observation in blue
                LOGGER.info(
                    "\n" +
                    "\033[1m" + f"[Worker {worker_id}] User:\n" + "\033[0m" +
                    f"\033[94m{user_msg}\033[0m" + "\n"
                )
        num_steps += 1

    LOGGER.info(
        f"[Worker {worker_id}] Task finished in {num_steps} steps. Success: {state.success}"
    )

    return state


def run_single_task(
    task: tasks.Task,
    agent_config: Dict[str, Any],
    task_config: Dict[str, Any],
    feedback_config: Dict[str, Any],
    env_config: Dict[str, Any],
    interactive_mode: bool,
    worker_id: int,
) -> Dict[str, Any]:
    """Run a single task in a worker thread."""
    # Each worker creates its own agent and tools instances
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )
    tools: List[Tool] = [
        getattr(importlib.import_module(module), class_name)()
        for module, class_name in task_config["tool_imports"]
    ]
    
    state = interactive_loop(
        task, agent, tools, feedback_config, env_config, interactive_mode, worker_id
    )
    
    return {"state": state.to_dict(), "task": task.to_dict()}


def main(args: argparse.Namespace):
    with open(args.exp_config) as f:
        exp_config: Dict[str, Any] = json.load(f)

    DEFAULT_FEEDBACK_CONFIG = exp_config["feedback_config"]
    DEFAULT_ENV_CONFIG = exp_config["env_config"]

    LOGGER.info(f"Experiment config: {exp_config}")

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    todo_tasks, n_tasks = task_class.load_tasks(task_config["filepath"])

    # initialize the feedback agent (if exist)
    feedback_config: Dict[str, Any] = exp_config.get(
        "feedback", DEFAULT_FEEDBACK_CONFIG
    )

    env_config: Dict[str, Any] = exp_config.get(
        "environment", DEFAULT_ENV_CONFIG)

    pathlib.Path(exp_config["output_dir"]).mkdir(parents=True, exist_ok=True)
    if args.interactive:
        output_path = os.path.join(
            exp_config["output_dir"], "results.interactive.jsonl")
    else:
        output_path = os.path.join(exp_config["output_dir"], "results.jsonl")

    done_task_id = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                task_id = json.loads(line)["task"].get("task_id", "")
                if task_id == "":
                    task_id = json.loads(line)["task"].get("id", "")
                done_task_id.add(task_id)
        LOGGER.info(
            f"Existing output file found. {len(done_task_id)} tasks done.")

    if len(done_task_id) == n_tasks:
        LOGGER.info("All tasks done. Exiting.")
        return

    # Filter out already done tasks
    remaining_tasks = [t for t in todo_tasks if t.task_id not in done_task_id]
    
    # Debug mode: only run first 3 tasks
    if args.debug:
        remaining_tasks = remaining_tasks[:3]
    
    n_tasks = len(remaining_tasks)
    
    # run the loop for all tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks with {args.num_workers} workers.")
    
    # Thread-safe lock for file writing
    write_lock = Lock()
    completed_count = [0]  # Use list to make it mutable in nested function
    
    def write_result(result: Dict[str, Any]):
        """Thread-safe result writer."""
        with write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            completed_count[0] += 1
            pbar.update(1)
    
    with logging_redirect_tqdm():
        pbar = tqdm(total=n_tasks)
        
        if args.num_workers > 1:
            # Parallel execution with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                # Submit all tasks
                future_to_task = {}
                for i, task in enumerate(remaining_tasks):
                    future = executor.submit(
                        run_single_task,
                        task,
                        exp_config["agent"],
                        task_config,
                        feedback_config,
                        env_config,
                        args.interactive,
                        i % args.num_workers,  # worker_id
                    )
                    future_to_task[future] = task
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        write_result(result)
                    except Exception as e:
                        LOGGER.error(f"Task {task.task_id} failed: {e}")
                        LOGGER.exception(e)
                        # Write error result
                        error_result = {
                            "state": {"finished": True, "success": False, "error": str(e)},
                            "task": task.to_dict()
                        }
                        write_result(error_result)
        else:
            # Sequential execution (original behavior)
            agent_config: Dict[str, Any] = exp_config["agent"]
            agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
                agent_config["config"]
            )
            tools: List[Tool] = [
                getattr(importlib.import_module(module), class_name)()
                for module, class_name in task_config["tool_imports"]
            ]
            
            for task in remaining_tasks:
                state = interactive_loop(
                    task, agent, tools, feedback_config, env_config, args.interactive
                )
                result = {"state": state.to_dict(), "task": task.to_dict()}
                write_result(result)
        
        pbar.close()
    
    LOGGER.info(f"Completed {completed_count[0]} tasks. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_config",
        type=str,
        default="./configs/gpt-3.5-turbo-0613/F=gpt-3.5-turbo-16k-0613/PHF=GT-textual/max5_p2+tool+cd/reasoning/scienceqa.json",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to run in interactive mode for demo purpose.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for task execution. Default is 1 (sequential).",
    )
    args = parser.parse_args()
    LOGGER.setLevel(logging.DEBUG if args.debug else logging.INFO)
    main(args)
