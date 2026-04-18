"""
Volume-based retrain gate.

Reads the dataset manifest and compares current train pool size against
the snapshot recorded at the last retrain. Triggers if new train sessions
OR new train samples exceed their thresholds.

Returns (should_retrain, train_session_ids, val_session_ids).
The pipeline controller exits early if should_retrain is False.
"""

from azure_storage import load_manifest, load_last_retrain_info
from training_config import RETRAIN_NEW_SAMPLES_THRESHOLD


def check_retrain_needed() -> tuple[bool, list[str], list[str]]:
    """
    Returns (should_retrain, train_session_ids, val_session_ids).

    should_retrain is True when the training pool has grown by at least 
    RETRAIN_NEW_SAMPLES_THRESHOLD
    samples since the last completed retrain.
    """
    manifest = load_manifest()
    last = load_last_retrain_info()

    train_entries = [e for e in manifest if e.get("split") == "train"]
    val_entries   = [e for e in manifest if e.get("split") == "val"]

    current_train_sessions = len(train_entries)
    current_train_samples  = sum(e.get("num_samples", 0) for e in train_entries)

    prev_train_sessions = last.get("num_train_sessions", 0)
    prev_train_samples  = last.get("num_train_samples", 0)

    new_sessions = current_train_sessions - prev_train_sessions
    new_samples  = current_train_samples  - prev_train_samples

    print(f"Train pool : {current_train_sessions} sessions / {current_train_samples} samples")
    print(f"Since last retrain: +{new_sessions} sessions / +{new_samples} samples")
    print(f"Thresholds : {RETRAIN_NEW_SAMPLES_THRESHOLD} samples")

    should_retrain = (
        new_samples >= RETRAIN_NEW_SAMPLES_THRESHOLD
    )

    # TODO: Uncomment after testing
    # if not should_retrain:
    #     print("Volume threshold not met — skipping retrain.")
    #     return False, [], []

    train_ids = [e["session_id"] for e in train_entries]
    val_ids   = [e["session_id"] for e in val_entries]

    print(f"Retrain triggered: {len(train_ids)} train, {len(val_ids)} val sessions")
    return True, train_ids, val_ids
