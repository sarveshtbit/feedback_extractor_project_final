import json
import os

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_cer(pred: str, gt: str) -> float:
    """Calculates Character Error Rate (CER) = Levenshtein Distance / len(gt)."""
    if not gt:
        return 0.0 if not pred else 1.0
    dist = levenshtein_distance(pred, gt)
    return dist / len(gt)

def evaluate_results(results_json_path: str):
    if not os.path.exists(results_json_path):
        print(f"File not found: {results_json_path}")
        return
    
    with open(results_json_path, "r") as f:
        data = json.load(f)
        
    print("=== EVALUATION AGAINST VERIFIED GROUND TRUTH ===")
    
    # Ground truth mapping for mobile numbers from cnn_dataset/ground_truth.csv
    gt_mobile = {
        "form1.jpeg": "9432164321",
        "form2.jpeg": "9042642526",
        "form3.jpeg": "8088997798",
        "form4.jpeg": "9754608192",
        "form5.jpeg": "8304934091",
    }
    
    variants = ["original", "tight_crop", "contrast_enhanced", "grid_removed", "contrast_plus_grid_removed"]
    
    stats = {v: {"total_chars": 0, "total_dist": 0, "exact_matches": 0, "total_fields": 0} for v in variants}
    
    for entry in data:
        fname = entry.get("filename")
        if fname not in gt_mobile:
            continue
        gt = gt_mobile[fname]
        mobile_data = entry.get("fields", {}).get("student_mobile", {})
        var_data = mobile_data.get("variants", {})
        
        print(f"\nFile: {fname} (GT: {gt[:2]}***{gt[-2:]})")
        for v in variants:
            pred_raw = var_data.get(v, {}).get("raw_text", "").replace(" ", "")
            # Filter digits only for mobile
            pred_digits = "".join([c for c in pred_raw if c.isdigit()])
            dist = levenshtein_distance(pred_digits, gt)
            cer = dist / len(gt)
            is_exact = (pred_digits == gt)
            
            stats[v]["total_chars"] += len(gt)
            stats[v]["total_dist"] += dist
            stats[v]["total_fields"] += 1
            if is_exact:
                stats[v]["exact_matches"] += 1
                
            masked_pred = pred_digits[:2] + "***" + pred_digits[-2:] if len(pred_digits) > 4 else "***"
            print(f"  Variant '{v:26}': Pred='{masked_pred}' (raw: '{pred_raw[:12]}'), CER={cer:.2f}, Exact={is_exact}")

    print("\n=== SUMMARY ACROSS ALL VERIFIED SAMPLES ===")
    for v in variants:
        st = stats[v]
        overall_cer = st["total_dist"] / max(1, st["total_chars"])
        exact_acc = (st["exact_matches"] / max(1, st["total_fields"])) * 100.0
        print(f"Variant: {v:28} | CER: {overall_cer:.4f} ({overall_cer*100:.1f}%) | Exact Match: {st['exact_matches']}/{st['total_fields']} ({exact_acc:.1f}%)")

if __name__ == '__main__':
    evaluate_results("storage/stage7_investigation/stage7_experiment_results.json")
