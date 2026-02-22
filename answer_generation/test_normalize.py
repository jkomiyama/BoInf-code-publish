#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正規化関数のテストスクリプト
"""
import sys
sys.path.append('/workspace/BoInf-code-publish_iclr2026/answer_generation')
from BoN_answeranalyze import normalize_math_notation

# テストケース
test_cases = [
    ("3", "3への正規化"),
    ("N=3", "N=3は3に"),
    ("c_{\\min}=2", "c_{\\min}=2は2に"),
    ("\\frac{1}{2}", "fracの処理"),
    ("\\lfrac{1}{2}", "lfracはfracと同じに"),
    ("\\Bigl(\\frac{1}{2}\\Bigr)", "Bigl/Bigrの削除"),
    ("a \\equiv b", "equivは=に"),
    ("2 \\cdot 3", "cdotはスペース（削除）に"),
    ("x_{\\max}=10", "x_{max}=10は10に"),
]

print("=" * 60)
print("正規化関数テスト")
print("=" * 60)

for test_input, description in test_cases:
    result = normalize_math_notation(test_input)
    print(f"\n入力: {test_input}")
    print(f"説明: {description}")
    print(f"出力: {result}")
    print("-" * 40)

print("\n" + "=" * 60)
print("テスト完了")
print("=" * 60)
