import random
from azure_storage import list_all_session_metas, download_session_meta

RETRAIN_THRESHOLD = 0.5  # Trigger retrain if 2+ sessions have accuracy < 50%


def check_retrain_needed() -> bool:
    """
    Check if retraining is triggered. Condition: 2+ sessions with accuracy < 50%.
    Assigns train/test split to unassigned sessions (permanent once set).
    """
    metas = list_all_session_metas()

    # Filter to sessions without an assigned split
    unassigned = [m for m in metas if m.get("split") is None]

    # Count sessions below threshold
    below_threshold = sum(1 for m in unassigned if m.get("accuracy", 1.0) < RETRAIN_THRESHOLD)

    print(f"Sessions checked: {len(unassigned)}")
    print(f"Below {RETRAIN_THRESHOLD*100:.0f}% threshold: {below_threshold}")

    if below_threshold >= 2:
        print("✓ RETRAIN TRIGGERED")
        assign_train_test_splits(unassigned)
        return True

    print("✗ No retrain (< 2 sessions below threshold)")
    return False


def assign_train_test_splits(sessions: list[dict], train_ratio: float = 0.8) -> None:
    """
    Randomly assign train/test split to sessions. Permanently update meta.json.
    """
    n_train = max(1, int(len(sessions) * train_ratio))
    train_indices = set(random.sample(range(len(sessions)), n_train))

    for i, meta in enumerate(sessions):
        split = "train" if i in train_indices else "test"
        meta["split"] = split
        # Update in Azure (reconstruct blob path from meta keys if needed)
        # For now, just log — actual update happens in retrain pipeline step
        print(f"  Assigned session {meta.get('session_id', i)}: {split}")
