"""
Eval gate: checks that the retriever matches denial descriptions to the
CORRECT synthetic rule, not just *a* rule. This is the accuracy claim
that should back up any README number — run it yourself, it's not asserted.

Usage: python evals/run_evals.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rules_engine import RuleRetriever

THRESHOLD = 0.85  # fraction of test cases that must match the correct rule_id

TEST_CASES = [
    {"text": "MRI was performed without an approved prior authorization on file.", "expected": "PA-001"},
    {"text": "The claim lacked clinical documentation tying the diagnosis to the procedure billed.", "expected": "MN-014"},
    {"text": "Claim was filed 200 days after the date of service, well past the deadline.", "expected": "TF-007"},
    {"text": "Patient saw an out-of-network specialist for a non-urgent consultation.", "expected": "NC-022"},
    {"text": "This exact claim, same provider and date, was already paid last month.", "expected": "DC-031"},
    {"text": "The procedure code billed was bundled into a more comprehensive code on the same day.", "expected": "CD-045"},
    {"text": "Eligibility records show the patient's plan was not active on the service date.", "expected": "EL-052"},
    {"text": "The plan requires trying a generic medication before covering the brand-name drug requested.", "expected": "RX-060"},
]


def main():
    retriever = RuleRetriever()
    correct = 0
    results = []

    for case in TEST_CASES:
        matches = retriever.retrieve(case["text"], top_k=1)
        predicted = matches[0].rule_id if matches else None
        is_correct = predicted == case["expected"]
        correct += int(is_correct)
        results.append({
            "text": case["text"],
            "expected": case["expected"],
            "predicted": predicted,
            "similarity": matches[0].similarity if matches else 0,
            "correct": is_correct,
        })

    accuracy = correct / len(TEST_CASES)
    print(json.dumps(results, indent=2))
    print(f"\nAccuracy: {correct}/{len(TEST_CASES)} = {accuracy:.1%} (threshold: {THRESHOLD:.0%})")

    if accuracy < THRESHOLD:
        print("EVAL GATE FAILED.")
        sys.exit(1)
    print("EVAL GATE PASSED.")


if __name__ == "__main__":
    main()
