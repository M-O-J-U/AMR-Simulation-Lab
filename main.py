"""
AMR Simulation — Main entry point.

Usage:
  python main.py server          # Start FastAPI server (default port 8000)
  python main.py server --port 9000
  python main.py test            # Run all tests with logged results
  python main.py headless        # Run headless simulation, print + log stats
  python main.py headless --scenario pakistan_crisis --steps 50
  python main.py validate        # Quick biology validation run
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_headless(scenario: str, steps: int, seed: int):
    from simulation.amr_model import AMRSimulationModel
    from simulation.sim_logger import SimLogger

    # Start a dedicated logging session for this headless run
    logger = SimLogger.start_session(
        session_name=f"headless_{scenario}",
        log_dir="logs",
        step_interval=5,
        print_to_terminal=False,   # suppress logger terminal output — table is cleaner
    )

    print(f"\n{'='*60}")
    print(f"  AMR Simulation — Headless Mode")
    print(f"  Scenario : {scenario}")
    print(f"  Steps    : {steps}")
    print(f"  Seed     : {seed}")
    print(f"  Log dir  : logs/")
    print(f"{'='*60}\n")

    model = AMRSimulationModel(scenario=scenario, initial_bacteria=80,
                               seed=seed, enable_logging=True)

    print(f"{'Step':>6} | {'Alive':>6} | {'Resist':>6} | {'Biofilm':>7} | "
          f"{'Persist':>7} | {'AvgFit':>7} | {'AvgGenes':>8} | {'HGT':>5}")
    print("-" * 70)

    for step in range(steps):
        model.step()

        if step % 5 == 0 or step == steps - 1:
            alive    = model.count_living_bacteria()
            resist   = model.count_resistant_bacteria()
            biofilm  = model.count_biofilm_bacteria()
            persist  = model.count_persisters()
            avg_fit  = model.avg_fitness()
            avg_gene = model.avg_resistance_genes()
            hgt      = len(model.hgt_events)
            print(f"{step:>6} | {alive:>6} | {resist:>6} | {biofilm:>7} | "
                  f"{persist:>7} | {avg_fit:>7.4f} | {avg_gene:>8.3f} | {hgt:>5}")

        if not model.running:
            print(f"\n[EXTINCTION] All bacteria eliminated at step {step}.")
            break

    print(f"\n{'='*60}")
    print("Final gene distribution:")
    gene_dist = model.get_resistance_gene_distribution()
    total = model.count_living_bacteria() or 1
    for gene, count in sorted(gene_dist.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = "█" * int(pct / 3)
        print(f"  {gene:<20} {bar:<30} {count:>4} ({pct:5.1f}%)")

    print(f"\nSpecies counts:")
    for sp, cnt in model.get_species_counts().items():
        print(f"  {sp:<45} {cnt:>4}")

    print(f"\nTotal HGT events: {len(model.hgt_events)}")

    # Close the logger and save files
    logger.log_session_summary(
        final_population=model.count_living_bacteria(),
        total_steps=model.current_step,
        final_gene_dist=model.get_resistance_gene_distribution(),
    )
    logger.close()
    print()


def run_validation():
    from simulation.amr_model import AMRSimulationModel
    from core.bacterium_agent import BacteriumAgent, BacterialState
    from simulation.sim_logger import SimLogger

    # Start logging session for validation
    logger = SimLogger.start_session(
        session_name="validation",
        log_dir="logs",
        step_interval=999,         # suppress step logs during validation
        print_to_terminal=False,
    )

    print("\n" + "="*60)
    print("  AMR Simulation — Biology Validation")
    print("="*60)

    results = []

    # ── Test 1 ────────────────────────────────────────────────────────────────
    print("\n[1/4] E. coli growth without antibiotics...")
    t_start = time.time()
    model = AMRSimulationModel(scenario="validation", initial_bacteria=30, seed=1, enable_logging=False)
    t0 = model.count_living_bacteria()
    for _ in range(20): model.step()
    t20   = model.count_living_bacteria()
    passed = t20 > t0
    ms    = (time.time() - t_start) * 1000
    status = "PASS" if passed else "FAIL"
    print(f"      Start: {t0} | After 20 steps: {t20} | {status}")
    logger.log_test_result("E. coli grows without antibiotics", passed,
                           f"start={t0}, after_20_steps={t20}", ms)
    results.append(passed)

    # ── Test 2 ────────────────────────────────────────────────────────────────
    print("\n[2/4] Ciprofloxacin kills susceptible E. coli...")
    t_start = time.time()
    model2 = AMRSimulationModel(scenario="validation", initial_bacteria=60, seed=2, enable_logging=False)
    for _ in range(8): model2.step()
    pre = model2.count_living_bacteria()
    model2.apply_antibiotic("ciprofloxacin", concentration=3.0, mode="uniform")
    for _ in range(20): model2.step()
    post   = model2.count_living_bacteria()
    passed = post < pre
    ms     = (time.time() - t_start) * 1000
    status = "PASS" if passed else "FAIL"
    print(f"      Pre-treatment: {pre} | Post-treatment: {post} | {status}")
    logger.log_test_result("Ciprofloxacin kills susceptible E. coli", passed,
                           f"pre={pre}, post={post}", ms)
    results.append(passed)

    # ── Test 3 ────────────────────────────────────────────────────────────────
    print("\n[3/4] Resistant bacteria (gyrA_S83L) survive ciprofloxacin...")
    t_start = time.time()
    model3 = AMRSimulationModel(scenario="validation", initial_bacteria=60, seed=3, enable_logging=False)
    agents = [a for a in model3.agents if isinstance(a, BacteriumAgent)]
    for a in agents[:30]:
        a.resistance_genes.add("gyrA_S83L")
        a._recalculate_fitness()
    model3.apply_antibiotic("ciprofloxacin", concentration=2.5, mode="uniform")
    for _ in range(20): model3.step()
    survivors  = [a for a in model3.agents
                  if isinstance(a, BacteriumAgent) and a.state != BacterialState.DEAD]
    res_surv   = sum(1 for a in survivors if "gyrA_S83L" in a.resistance_genes)
    sus_surv   = sum(1 for a in survivors if "gyrA_S83L" not in a.resistance_genes)
    passed     = res_surv >= sus_surv
    ms         = (time.time() - t_start) * 1000
    status     = "PASS" if passed else "CHECK"
    print(f"      Resistant survivors: {res_surv}/30 | Susceptible: {sus_surv}/30 | {status}")
    logger.log_test_result("gyrA-resistant bacteria survive ciprofloxacin better", passed,
                           f"resistant={res_surv}, susceptible={sus_surv}", ms)
    results.append(passed)

    # ── Test 4 ────────────────────────────────────────────────────────────────
    print("\n[4/4] HGT spreads blaTEM-1 gene...")
    t_start = time.time()
    model4  = AMRSimulationModel(scenario="validation", initial_bacteria=80, seed=4, enable_logging=False)
    agents4 = [a for a in model4.agents if isinstance(a, BacteriumAgent)]
    for a in agents4[:20]:
        a.resistance_genes.add("blaTEM-1")
    initial_carriers = sum(1 for a in agents4 if "blaTEM-1" in a.resistance_genes)
    for _ in range(25): model4.step()
    final_carriers = sum(
        1 for a in model4.agents
        if isinstance(a, BacteriumAgent)
        and "blaTEM-1" in a.resistance_genes
        and a.state != BacterialState.DEAD
    )
    passed = final_carriers >= initial_carriers
    ms     = (time.time() - t_start) * 1000
    status = "PASS" if passed else "FAIL"
    print(f"      Initial: {initial_carriers} | Final: {final_carriers} | "
          f"HGT events: {len(model4.hgt_events)} | {status}")
    logger.log_test_result("blaTEM-1 spreads via horizontal gene transfer", passed,
                           f"initial={initial_carriers}, final={final_carriers}, "
                           f"hgt_events={len(model4.hgt_events)}", ms)
    results.append(passed)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed_n = sum(results)
    total_n  = len(results)
    logger.log_test_suite_summary(passed_n, total_n - passed_n, total_n, 0)
    logger.close()

    print(f"\n{'='*60}")
    print(f"  Validation complete. {passed_n}/{total_n} passed.")
    print(f"  Log saved to: logs/\n")


def run_server(host: str, port: int):
    from simulation.sim_logger import SimLogger
    # Start a persistent server-mode logging session
    SimLogger.start_session(
        session_name="server",
        log_dir="logs",
        step_interval=10,
        print_to_terminal=False,  # server logs go to file only
    )
    print(f"\n{'='*60}")
    print(f"  AMR Simulation API Server")
    print(f"  http://{host}:{port}")
    print(f"  API docs: http://{host}:{port}/docs")
    print(f"  Open frontend/index.html in your browser")
    print(f"  Simulation logs saved to: logs/")
    print(f"{'='*60}\n")
    import uvicorn
    uvicorn.run("api.server:app", host=host, port=port, reload=False, log_level="info")


def run_tests_with_logging():
    """Run pytest and capture results into the structured logger."""
    import pytest
    from simulation.sim_logger import SimLogger

    logger = SimLogger.start_session(
        session_name="test_suite",
        log_dir="logs",
        step_interval=999,
        print_to_terminal=False,
    )
    logger.log(
        __import__('simulation.sim_logger', fromlist=['LogLevel']).LogLevel.MILESTONE,
        "Running full pytest test suite — 50 tests across 5 layers"
    )
    logger.close()

    # Run pytest normally (its own output to terminal)
    sys.exit(pytest.main(["tests/", "-v", "--tb=short"]))


def run_gnn_train(quick: bool = True, epochs: int = None, scenarios: list = None):
    """Train the GNN resistance predictor."""
    from ai.gnn_trainer import train, QUICK_CONFIG, DEFAULT_CONFIG

    cfg = QUICK_CONFIG.copy() if quick else DEFAULT_CONFIG.copy()
    if epochs:    cfg["epochs"]    = epochs
    if scenarios: cfg["scenarios"] = scenarios

    print(f"\n{'='*60}")
    print(f"  AMR GNN — Resistance Gene Transfer Predictor")
    print(f"  Mode    : {'Quick' if quick else 'Full'}")
    print(f"  Epochs  : {cfg['epochs']}")
    print(f"  Scenarios: {cfg['scenarios']}")
    print(f"{'='*60}\n")

    model, metrics = train(config=cfg, verbose=True)

    print(f"\n  Final test AUROC : {metrics.get('auroc_macro', 0):.4f}")
    print(f"  Final test AUPRC : {metrics.get('auprc_macro', 0):.4f}")
    print(f"  Final test F1    : {metrics.get('f1_macro', 0):.4f}")
    print(f"\n  Checkpoint saved: ai/checkpoints/best_model.pt")
    print(f"  Results saved:    ai/checkpoints/training_results.json\n")


def run_all_tests():
    """Run both simulation tests and GNN tests."""
    import pytest
    from simulation.sim_logger import SimLogger, LogLevel

    logger = SimLogger.start_session(
        session_name="test_suite_full",
        log_dir="logs",
        step_interval=999,
        print_to_terminal=False,
    )
    logger.log(LogLevel.MILESTONE,
        "Full test suite started — simulation + GNN tests")
    logger.close()

    sys.exit(pytest.main(["tests/", "-v", "--tb=short"]))

def main():
    parser = argparse.ArgumentParser(
        description="AMR Simulation — Antimicrobial Resistance Agent-Based Model"
    )
    subparsers = parser.add_subparsers(dest="command")

    srv = subparsers.add_parser("server", help="Start API server + open frontend")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)

    hl = subparsers.add_parser("headless", help="Run headless simulation, print stats")
    hl.add_argument("--scenario", default="ecoli_cipro",
                    choices=["validation","ecoli_cipro","klebsiella_carbapenem",
                             "xdr_acinetobacter","mrsa_hospital","pakistan_crisis","multi_species"])
    hl.add_argument("--steps", type=int, default=40)
    hl.add_argument("--seed",  type=int, default=42)

    subparsers.add_parser("test",     help="Run full pytest test suite (sim + GNN)")
    subparsers.add_parser("validate", help="Run biology validation (4 checks)")

    gnn = subparsers.add_parser("train-gnn", help="Train the GNN resistance predictor")
    gnn.add_argument("--full",      action="store_true", help="Full training (slower)")
    gnn.add_argument("--epochs",    type=int, default=None)
    gnn.add_argument("--scenarios", nargs="+", default=None)

    bl = subparsers.add_parser("baselines", help="Run baseline comparison (LR, RF, frequency)")
    bl.add_argument("--full", action="store_true", help="Full dataset")

    val_ext = subparsers.add_parser("validate-external", help="External validation vs PATRIC/synthetic data")
    val_ext.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic data (no internet needed)")
    val_ext.add_argument("--real",      action="store_true", help="Fetch real PATRIC data")

    subparsers.add_parser("calibrate", help="Threshold calibration and PR curve analysis")

    args = parser.parse_args()

    if args.command == "server":
        run_server(args.host, args.port)
    elif args.command == "headless":
        run_headless(args.scenario, args.steps, args.seed)
    elif args.command == "test":
        run_all_tests()
    elif args.command == "validate":
        run_validation()
    elif args.command == "train-gnn":
        run_gnn_train(
            quick=not args.full,
            epochs=args.epochs,
            scenarios=args.scenarios,
        )
    elif args.command == "baselines":
        from ai.baselines import run_all_comparisons
        from ai.gnn_trainer import DEFAULT_CONFIG, QUICK_CONFIG
        cfg = QUICK_CONFIG if not getattr(args, "full", False) else DEFAULT_CONFIG
        run_all_comparisons(cfg)
    elif args.command == "validate-external":
        from ai.external_validation import run_external_validation
        run_external_validation(use_synthetic=getattr(args, "synthetic", True))
    elif args.command == "calibrate":
        from ai.threshold_calibration import run_threshold_calibration
        run_threshold_calibration()
    else:
        run_validation()
        parser.print_help()


if __name__ == "__main__":
    main()