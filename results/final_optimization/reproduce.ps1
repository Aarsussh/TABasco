$ErrorActionPreference = 'Stop'
$python = '..\.venv\Scripts\python.exe'

# Fresh 3090 screening runs (all reward settings specified in experiment_plan.md).
& $python results\final_optimization\train_variant.py --name lost_1p5 --seed 3090 --lost 1.5
& $python results\final_optimization\train_variant.py --name restore_0p6 --seed 3090 --restore 0.6
& $python results\final_optimization\train_variant.py --name disrupt_0p35 --seed 3090 --disruption 0.35
& $python results\final_optimization\train_variant.py --name balanced --seed 3090 --lost 1.5 --restore 0.7 --disruption 0.4

# Confirmatory second seeds for the two validation-screen leaders.
& $python results\final_optimization\train_variant.py --name lost_1p5 --seed 4090 --lost 1.5
& $python results\final_optimization\train_variant.py --name disrupt_0p35 --seed 4090 --disruption 0.35

# Common 40-seed validation. Control and candidates all use 92000-92039.
& $python evaluate.py --checkpoint 3090=models/sensitivity/constrained_seed3090.pt --checkpoint 4090=models/sensitivity/constrained_seed4090.pt --episodes 40 --seed 92000 --constraint-aware --results-dir results/final_optimization/control_validation
& $python results\final_optimization\evaluate_variant.py --name lost_1p5 --lost 1.5 --checkpoint 3090=results/final_optimization/checkpoints/lost_1p5_seed3090.pt --out results/final_optimization/screen_lost_1p5
& $python results\final_optimization\evaluate_variant.py --name restore_0p6 --restore 0.6 --checkpoint 3090=results/final_optimization/checkpoints/restore_0p6_seed3090.pt --out results/final_optimization/screen_restore_0p6
& $python results\final_optimization\evaluate_variant.py --name disrupt_0p35 --disruption 0.35 --checkpoint 3090=results/final_optimization/checkpoints/disrupt_0p35_seed3090.pt --out results/final_optimization/screen_disrupt_0p35
& $python results\final_optimization\evaluate_variant.py --name balanced --lost 1.5 --restore 0.7 --disruption 0.4 --checkpoint 3090=results/final_optimization/checkpoints/balanced_seed3090.pt --out results/final_optimization/screen_balanced
& $python results\final_optimization\evaluate_variant.py --name lost_1p5_confirmatory --lost 1.5 --checkpoint 3090=results/final_optimization/checkpoints/lost_1p5_seed3090.pt --checkpoint 4090=results/final_optimization/checkpoints/lost_1p5_seed4090.pt --out results/final_optimization/confirm_lost_1p5
& $python results\final_optimization\evaluate_variant.py --name disrupt_0p35_confirmatory --disruption 0.35 --checkpoint 3090=results/final_optimization/checkpoints/disrupt_0p35_seed3090.pt --checkpoint 4090=results/final_optimization/checkpoints/disrupt_0p35_seed4090.pt --out results/final_optimization/confirm_disrupt_0p35

& $python results\final_optimization\summarize_optimization.py
& $python results\final_optimization\plot_optimization.py
