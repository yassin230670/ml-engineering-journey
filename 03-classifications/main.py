import argparse
from pathlib import Path
import pandas as pd
from src.data_preprocessing import load_iris_csv, SPECIES
from src.experiments import train_binary_experiment, run_ovr, save_comparison


def main():
    parser = argparse.ArgumentParser(description="Perceptron from scratch: Iris experiments")
    parser.add_argument("--data", default="data/Iris.csv", help="Path to Iris CSV file")
    parser.add_argument("--results", default="results", help="Directory for generated plots and reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = load_iris_csv(args.data)
    print(f"Loaded {len(df)} samples.")
    print("Species counts:\n", df["Species"].value_counts().to_string())

    result1 = train_binary_experiment(
        df, "Iris-setosa", "Iris-versicolor", "experiment1_setosa_vs_versicolor",
        results_dir, seed=args.seed
    )
    result2 = train_binary_experiment(
        df, "Iris-versicolor", "Iris-virginica", "experiment2_versicolor_vs_virginica",
        results_dir, seed=args.seed
    )
    result3 = run_ovr(df, results_dir, seed=args.seed)
    save_comparison([result1, result2, result3], results_dir)
    print(f"\nDone. Plots and reports saved to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()