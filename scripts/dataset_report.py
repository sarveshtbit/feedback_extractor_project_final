import os
import sys
sys.path.insert(0, os.getcwd())
import pandas as pd
from collections import defaultdict
from backend.app.ml.training.dataset_manager import DatasetManager

def generate_dataset_report(dataset_dir: str = "real_digit_dataset"):
    print("==================================================")
    print("REAL HANDWRITTEN DATASET & SPLIT REPORT")
    print("==================================================")

    dm = DatasetManager(dataset_dir)
    data_info = dm.validate_and_parse_dataset()

    if not data_info.get("valid"):
        print(f"Error loading dataset: {data_info.get('error')}")
        return

    samples = data_info["samples"]
    total_samples = len(samples)

    # Forms & Writers
    forms = sorted(list(set([s.get("original_form_filename", s["writer_id"]) for s in samples])))
    writers = sorted(list(set([s["writer_id"] for s in samples])))

    # Samples per class
    class_counts = defaultdict(int)
    position_counts = defaultdict(int)
    writer_counts = defaultdict(int)

    for s in samples:
        class_counts[s["label"]] += 1
        position_counts[s.get("digit_position", "?")] += 1
        writer_counts[s["writer_id"]] += 1

    # Writer-independent splits
    splits = dm.create_writer_independent_splits()
    split_summary = splits["split_summary"]
    reliability_notice = splits.get("reliability_notice", "INSUFFICIENT WRITERS FOR RELIABLE GENERALIZATION EVALUATION")

    print(f"\n1. SUMMARY METRICS:")
    print(f"   • Total Forms:            {len(forms)} ({', '.join(forms)})")
    print(f"   • Total Writers:          {len(writers)} ({', '.join(writers)})")
    print(f"   • Total Digit Crops:      {total_samples}")
    print(f"   • Reliability Status:     [{reliability_notice}]")

    print(f"\n2. SAMPLES PER DIGIT CLASS (0-9):")
    for digit in range(10):
        print(f"   • Digit {digit}: {class_counts[digit]} samples")

    print(f"\n3. SAMPLES PER WRITER/FORM:")
    for w in writers:
        print(f"   • {w}: {writer_counts[w]} samples")

    print(f"\n4. SAMPLES PER DIGIT POSITION (P1-P10):")
    for pos in range(1, 11):
        print(f"   • Position P{pos:02d}: {position_counts[pos]} samples")

    print(f"\n5. WRITER-INDEPENDENT SPLIT ALLOCATION:")
    print(f"   • Train Writers ({len(split_summary['train_writers'])}):      {split_summary['train_writers']} ({split_summary['train_samples_count']} samples)")
    print(f"   • Validation Writers ({len(split_summary['val_writers'])}): {split_summary['val_writers']} ({split_summary['val_samples_count']} samples)")
    print(f"   • Test Writers ({len(split_summary['test_writers'])}):       {split_summary['test_writers']} ({split_summary['test_samples_count']} samples)")

    # Class balance across splits
    def get_class_dist(sample_list):
        dist = defaultdict(int)
        for s in sample_list:
            dist[s["label"]] += 1
        return {d: dist[d] for d in range(10)}

    print(f"\n6. CLASS DISTRIBUTION ACROSS SPLITS:")
    print(f"   • Train Class Dist: {get_class_dist(splits['train'])}")
    print(f"   • Val Class Dist:   {get_class_dist(splits['val'])}")
    print(f"   • Test Class Dist:  {get_class_dist(splits['test'])}")

    print(f"\n7. GENERALIZATION EVALUATION NOTICE:")
    if len(writers) < 10:
        print(f"   [!] WARNING: {reliability_notice}")
        print(f"   Current dataset metrics are NOT STATISTICALLY RELIABLE FOR GENERALIZATION.")
    else:
        print(f"   [+] Dataset is statistically sufficient for split generalization evaluation.")

    print("=" * 60)

if __name__ == "__main__":
    generate_dataset_report()
