"""
EyeSist base model retrain pipeline.

Runs on a weekly ClearML schedule. The check_retrain step acts as the gate —
if the training pool hasn't grown enough since the last retrain, the pipeline
exits early without training. Can also be triggered manually at any time.

Pipeline steps:
  1. check_retrain   — volume-based gate; assigns 70/15/15 splits via greedy balancer
  2. train_model     — retrain from production weights on full train split
  3. evaluate_promote — compare candidate vs production on frozen cumulative test set

Usage:
  Register weekly schedule (run once to activate):
    python pipeline_controller.py --schedule

  Trigger immediately (manual override, bypasses schedule):
    python pipeline_controller.py --run-now

  Local debug (all steps run in this process, no ClearML agents needed):
    python pipeline_controller.py --run-local
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from clearml import Task
from clearml.automation import PipelineDecorator

CLEARML_PROJECT = "EyeSist"
PIPELINE_NAME = "Base Model Retrain"
EXECUTION_QUEUE = "default"
WEEKLY_CRON = "0 2 * * 1"  # 2am UTC every Monday


# ── Pipeline components ────────────────────────────────────────────────────────

@PipelineDecorator.component(
    name="check_retrain",
    return_values=["should_retrain", "train_session_ids", "val_session_ids"],
    task_type=Task.TaskTypes.data_processing,
    execution_queue=EXECUTION_QUEUE,
    packages=["azure-identity", "azure-storage-blob"],
)
def component_check_retrain() -> tuple[bool, list, list]:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.step_check_retrain import check_retrain_needed
    return check_retrain_needed()


@PipelineDecorator.component(
    name="train_model",
    return_values=["candidate_blob", "val_accuracy"],
    task_type=Task.TaskTypes.training,
    execution_queue=EXECUTION_QUEUE,
    packages=[
        "azure-identity", "azure-storage-blob",
        "torch", "torchvision", "pillow", "numpy", "scikit-learn",
    ],
)
def component_train_model(
    train_session_ids: list, val_session_ids: list
) -> tuple[str, float]:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.step_train_base_model import run_training
    result = run_training(
        train_session_ids=train_session_ids,
        val_session_ids=val_session_ids,
    )
    return result["candidate_blob"], result["val_accuracy"]


@PipelineDecorator.component(
    name="evaluate_promote",
    return_values=["promoted", "candidate_test_acc"],
    task_type=Task.TaskTypes.qc,
    execution_queue=EXECUTION_QUEUE,
    packages=[
        "azure-identity", "azure-storage-blob",
        "torch", "torchvision", "pillow", "numpy",
    ],
)
def component_evaluate_promote(
    candidate_blob: str, candidate_val_accuracy: float
) -> tuple[bool, float]:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.step_evaluate_promote import evaluate_and_promote
    result = evaluate_and_promote(candidate_blob, candidate_val_accuracy)
    return result["promoted"], result["candidate_test_acc"]


# ── Pipeline definition ────────────────────────────────────────────────────────

@PipelineDecorator.pipeline(
    name=PIPELINE_NAME,
    project=CLEARML_PROJECT,
    version="1.0",
    pipeline_execution_queue=EXECUTION_QUEUE,
    add_pipeline_tags=True,
)
def retrain_pipeline() -> dict:
    should_retrain, train_ids, val_ids = component_check_retrain()

    if not should_retrain:
        print("Volume threshold not met — no retrain needed this week.")
        return {"status": "skipped"}

    candidate_blob, val_accuracy = component_train_model(train_ids, val_ids)
    promoted, test_accuracy = component_evaluate_promote(candidate_blob, val_accuracy)

    print(
        f"Pipeline complete | promoted={promoted} | "
        f"candidate_test_acc={test_accuracy:.4f}"
    )
    return {"promoted": promoted, "candidate_test_acc": test_accuracy}


# ── Schedule registration ──────────────────────────────────────────────────────

def register_schedule(cron: str = WEEKLY_CRON) -> None:
    """
    Register this pipeline as a weekly ClearML scheduled job.
    Run this once; the ClearML scheduler service handles all future executions.
    To change the schedule, re-run with a different --cron value.
    """
    from clearml.automation.scheduler import TaskScheduler

    parts = cron.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5-field cron expression, got: '{cron}'")

    minute_s, hour_s, day_s, month_s, weekday_s = parts
    if day_s != "*" or month_s != "*":
        raise ValueError(
            "This ClearML scheduler wrapper only supports weekly cron expressions "
            "with '*' for day-of-month and month."
        )

    try:
        minute = int(minute_s)
        hour = int(hour_s)
    except ValueError as exc:
        raise ValueError(f"Invalid cron minute/hour in '{cron}'") from exc

    weekday_map = {
        "0": "sunday",
        "1": "monday",
        "2": "tuesday",
        "3": "wednesday",
        "4": "thursday",
        "5": "friday",
        "6": "saturday",
        "7": "sunday",
    }
    weekdays = [weekday_map.get(token.strip()) for token in weekday_s.split(",")]
    if any(day is None for day in weekdays):
        raise ValueError(
            "Unsupported cron weekday field. Use comma-separated numeric weekdays "
            "like '1' for Monday."
        )

    print(f"Registering weekly pipeline schedule (cron: '{cron}')")
    scheduler = TaskScheduler(
        sync_frequency_minutes=60,
    )
    scheduler.add_task(
        schedule_function=retrain_pipeline,
        queue=EXECUTION_QUEUE,
        name=f"{PIPELINE_NAME} — Weekly",
        minute=minute,
        hour=hour,
        weekdays=weekdays,
        target_project=CLEARML_PROJECT,
    )
    scheduler.start_remotely()
    print("Scheduler registered as a ClearML service task.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EyeSist retrain pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--schedule",
        action="store_true",
        help="Register weekly ClearML scheduler (run once to activate)",
    )
    mode.add_argument(
        "--run-now",
        action="store_true",
        help="Trigger the pipeline immediately on ClearML agents (manual override)",
    )
    mode.add_argument(
        "--run-local",
        action="store_true",
        help="Run all steps locally in this process (debug only)",
    )
    parser.add_argument(
        "--cron",
        default=WEEKLY_CRON,
        metavar="EXPR",
        help=f"Cron expression for --schedule (default: '{WEEKLY_CRON}' = 2am UTC Monday)",
    )
    args = parser.parse_args()

    if args.schedule:
        register_schedule(cron=args.cron)
    elif args.run_now:
        print("Triggering pipeline immediately on ClearML agents...")
        retrain_pipeline()
    elif args.run_local:
        PipelineDecorator.run_locally()
        retrain_pipeline()
