"""
Generate paper/table1_genes.tex (Resistance genes: CARD identifiers and
biological parameters) directly from the simulation's gene table, so the
table can never drift from the code that produced the results.

  python paper/make_table1.py

Uses the pipeline's biology (data.biology.PAPER_V1 = data/card_loader.py).
CARD IDs verified against card.mcmaster.ca on 2026-09-29.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.biology import PAPER_V1

MECHANISM = {  # short labels for the table (wording carried over from the retired draft)
    "blaTEM-1": r"$\beta$-lactamase", "blaCTX-M-15": "ESBL", "blaKPC-2": "Carbapenemase",
    "blaNDM-1": "Metallo-BL", "mexAB-oprM": "Efflux pump", "acrAB-tolC": "Efflux pump",
    "gyrA_S83L": "Target alteration", "mcr-1": "Polymyxin resist.",
    "tetM": "Ribosomal prot.", "vanA": "Glycopeptide resist.",
}
MARK = {"mexAB-oprM": "a", "acrAB-tolC": "a", "gyrA_S83L": "b", "mcr-1": "c"}
ORDER = ["blaTEM-1", "blaCTX-M-15", "blaKPC-2", "blaNDM-1", "mexAB-oprM",
         "acrAB-tolC", "gyrA_S83L", "mcr-1", "tetM", "vanA"]


def row(name):
    g = PAPER_V1.genes[name]
    aro = g.card_id.replace("ARO:", "")
    if name in MARK:
        aro += f"$^{{{MARK[name]}}}$"
    label = name.replace("_", r"\_")
    return (f"{label:<12}& {MECHANISM[name]:<20}& {aro:<16}& "
            f"{g.fitness_cost:.2f} & {g.acquisition_prob:.3f} \\\\")


def table() -> str:
    assert list(PAPER_V1.genes) == ORDER, "gene table changed; update ORDER/MECHANISM"
    return "\n".join([
        r"\begin{table}[!t]",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{Resistance Genes: CARD Identifiers and Biological Parameters}",
        r"\label{tab:genes}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"\textbf{Gene} & \textbf{Mechanism} & \textbf{CARD ARO} &",
        r"\textbf{Fit.\ cost} & \textbf{Acq./step} \\",
        r"\midrule",
        *[row(n) for n in ORDER],
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\footnotesize{$^{a}$CARD efflux-complex entries MexAB-OprM and AcrAB-TolC.",
        r"$^{b}$CARD entry ``\textit{Escherichia coli gyrA} conferring resistance to",
        r"fluoroquinolones''; S83L is one of the variants this model covers, not a",
        r"standalone ARO entry. $^{c}$CARD entry name MCR-1.1.}",
        r"\end{table}",
        "",
    ])


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table1_genes.tex")
    open(out, "w", encoding="utf-8").write(table())
    print(open(out, encoding="utf-8").read())
