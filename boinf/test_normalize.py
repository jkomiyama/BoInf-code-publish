#!/usr/bin/env python3
"""Test normalization for answerbench."""

from ensemble_utils import normalize_answer_format

# Test cases from the user's examples
test_cases = [
    ("2^n", "2^{n}"),  # Should match
    ("2x-2", "2x-2\\;"),  # Should match
    ("\\lfloor\\frac{a}{2}\\rfloor+1", "\\;\\lfloor\\frac{a}{2}\\rfloor+1"),  # Should match
    ("3", "3"),  # Exact match
]

print("=== Normalization tests ===")
for gold, pred in test_cases:
    norm_gold = normalize_answer_format(gold)
    norm_pred = normalize_answer_format(pred)
    match = norm_gold == norm_pred
    print(f"Gold: '{gold}' -> '{norm_gold}'")
    print(f"Pred: '{pred}' -> '{norm_pred}'")
    print(f"Match: {match}")
    print()

# Additional edge cases
additional_tests = [
    "2^{n}",
    "2^n",
    "\\lfloor\\frac{a}{2}\\rfloor+1",
    "\\;\\lfloor\\frac{a}{2}\\rfloor+1\\;",
    "2x-2\\;",
    "2x-2",
]

print("=== Additional normalization tests ===")
for test in additional_tests:
    normalized = normalize_answer_format(test)
    print(f"'{test}' -> '{normalized}'")
