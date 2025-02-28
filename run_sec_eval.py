import json
import logging
import os
import time
import glob
import argparse
import dotenv
from desktop_env.desktop_env import DesktopEnv
import lib_run_single
from mm_agents.computer_use import ComputerUseAgent
from dataclasses import dataclass
import colorlog

dotenv.load_dotenv()

# Configure colored formatter
colored_formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s - %(message)s',
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

# Configure root logger with colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colored_formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

# Configure agent logger
agent_logger = logging.getLogger("desktopenv.agent")
agent_logger.setLevel(logging.INFO)
agent_logger.propagate = False  # Prevent propagation to root logger
agent_logger.addHandler(handler)  # Use the same colored handler

# Fixed model to use for all runs
MODEL = "claude-3-5-sonnet-20241022"

@dataclass
class Args:
    sleep_after_execution: float = 1.0


def get_scenario_files():
    """Get all JSON scenario files from the sec_eval directory."""
    return glob.glob("sec_eval/*.json")


def run_scenario(scenario_path, num_runs):
    """Run a single scenario multiple times."""
    # Extract scenario name from file path
    scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
    
    # Load the scenario
    with open(scenario_path, "r") as f:
        scenario = json.loads(f.read())
    
    # Create results directory for this scenario
    scenario_result_dir = os.path.join(
        "results",
        scenario_name,
    )
    os.makedirs(scenario_result_dir, exist_ok=True)
    
    # Get scenario details
    instruction = scenario["instruction"]
    snapshot = scenario["snapshot"]
    
    # Store scenario scores
    all_scores = []
    
    # Log start of scenario
    root_logger.info(f"Starting scenario: {scenario_name}")
    root_logger.info(f"Planning to run {num_runs} times")
    
    # Run the scenario specified number of times
    for run_idx in range(num_runs):
        # Create timestamped directory for this run
        run_result_dir = os.path.join(
            scenario_result_dir,
            f"{time.strftime('%Y%m%d@%H%M%S')}_run_{run_idx + 1}"
        )
        os.makedirs(run_result_dir, exist_ok=True)
        
        # Create agent
        agent = ComputerUseAgent(
            model=MODEL,
            max_tokens=4096,
            max_trajectory_length=20,
            message_path=f"{run_result_dir}/messages.json"
        )
        
        # Scores for this run
        run_scores = []
        
        try:
            root_logger.info(f"Starting run {run_idx + 1}/{num_runs} for scenario: {scenario_name}")
            
            # Create environment
            env = DesktopEnv(action_space="pyautogui", require_terminal=True, snapshot_name=snapshot)
            
            # Run the example
            lib_run_single.run_single_example(
                agent,
                env,
                scenario,
                20,  # max_steps
                instruction,
                Args(),  # properly structured args object
                run_result_dir,
                run_scores,
            )
        except Exception as e:
            agent_logger.error(f"Exception in run {run_idx + 1}: {e}")
            env.controller.end_recording(
                os.path.join(run_result_dir, "recording.mp4")
            )
            with open(os.path.join(run_result_dir, "traj.jsonl"), "a") as f:
                f.write(
                    json.dumps(
                        {"Error": f"Exception occurred: {str(e)}"}
                    )
                )
                f.write("\n")
        finally:
            env.close()
            
            # Collect score if available
            if run_scores:
                all_scores.append(run_scores[0])
    
    return all_scores


def main():
    # Parse command line arguments - only allow configuring number of runs
    parser = argparse.ArgumentParser(description="Run security evaluation scenarios")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per scenario")
    args = parser.parse_args()
    
    # Get all scenario files
    all_scenarios = get_scenario_files()
    
    if not all_scenarios:
        root_logger.error("No scenarios found in sec_eval directory")
        return
    
    # Create results directory
    os.makedirs("sec_eval_results", exist_ok=True)
    
    # Log start of evaluation
    root_logger.info(f"Starting security evaluation with {len(all_scenarios)} scenarios")
    root_logger.info(f"Each scenario will be run {args.runs} times using model {MODEL}")
    
    # Run all scenarios
    results = {}
    for scenario_path in all_scenarios:
        scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
        results[scenario_name] = run_scenario(scenario_path, args.runs)
    
    # Log summary
    root_logger.info("Security evaluation complete")
    for scenario_name, scores in results.items():
        avg_score = sum(scores) / len(scores) if scores else None
        root_logger.info(f"Scenario {scenario_name}: average score = {avg_score}")


if __name__ == "__main__":
    main() 