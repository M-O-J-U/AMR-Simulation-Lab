"""paper/table1_genes.tex must match what paper/make_table1.py generates from
the current gene table (so the paper can never cite stale CARD IDs/params)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper"))
import make_table1


def test_table1_up_to_date():
    committed = open(os.path.join(ROOT, "paper", "table1_genes.tex"), encoding="utf-8").read()
    assert committed == make_table1.table(), "run: python paper/make_table1.py"
