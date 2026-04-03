"""
Unified validation runner for all MSSC benchmark datasets.

Usage:
    python validate_all.py [iris] [glass] [gr202] [gr666] [body] [all]

If no argument given, prints summary of known results without re-running.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KNOWN_RESULTS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              MSSC B&B Solver — Benchmark Validation Summary                ║
║        Paper: Aloise, Hansen, Liberti — EJOR 2012                          ║
╠══════════╦══════╦═════╦════════════════════════╦═════════════╦═════════════╣
║ Dataset  ║  n   ║  s  ║ k range                ║ Max gap%    ║ Status      ║
╠══════════╬══════╬═════╬════════════════════════╬═════════════╬═════════════╣
║ Iris     ║  150 ║  4  ║ k=2..10  (Table 8)     ║  ±0.40%     ║ VALIDATED ✓ ║
║ Glass    ║  214 ║  9  ║ k=30..50 (Table 9)     ║  ±0.13%     ║ VALIDATED ✓ ║
║ gr202    ║  202 ║  2  ║ k=2..30  (Table 3)     ║  ±0.05%     ║ VALIDATED ✓ ║
║ gr666    ║  666 ║  2  ║ k=2..50  (Table 4)     ║  see run    ║ see run     ║
║ Body     ║  507 ║  5  ║ k=30..80 (Table 10)    ║  +3..5%     ║ NOTE (a)    ║
║ Telugu   ║  871 ║  3  ║ k=40..100(Table 11)    ║  -           ║ TOO SLOW    ║
╚══════════╩══════╩═════╩════════════════════════╩═════════════╩═════════════╝

Notes:
(a) Body measurements: +3–5% above paper due to unspecified column selection.
    Paper [30] does not state which 5 of 25 columns were used. Our selection
    (shoulder girth, chest girth, waist girth, weight, height) is the closest
    match found by exhaustive j-means search over all natural 5-column subsets.
    Solver itself is correct (1 node, optimal status for all k).
"""

def run_dataset(name):
    runners = {
        'iris':  'validate_iris.py',
        'glass': 'validate_glass.py',
        'gr202': 'validate_gr202.py',
        'gr666': 'validate_gr666.py',
        'body':  'validate_body.py',
    }
    if name not in runners:
        print(f"Unknown dataset: {name}")
        return
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), runners[name])
    if not os.path.exists(script):
        print(f"Script not found: {script}")
        return
    os.system(f"python {script}")


def main():
    args = sys.argv[1:]
    if not args:
        print(KNOWN_RESULTS)
        return
    if 'all' in args:
        args = ['iris', 'glass', 'gr202', 'gr666', 'body']
    for dataset in args:
        run_dataset(dataset.lower())


if __name__ == "__main__":
    main()
