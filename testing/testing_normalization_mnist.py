import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

accuracies = []
norm_values = []

base_dir = os.path.dirname(os.path.abspath(__file__))  # testing folder
parent_dir = os.path.dirname(base_dir)                # parent folder (Project1)

script_path = os.path.join(parent_dir, "mnist_normalization_parameter_logistic_regression.py")
script_path = os.path.normpath(script_path)

csv_path = os.path.join(parent_dir, "mnist", "mnist_train.csv")
csv_path = os.path.normpath(csv_path)

norm_test_values = np.arange(1, 255, 25)

for norm_val in norm_test_values:
    print(f"Running: python {script_path} {csv_path} 0.2 {norm_val} 10000")

    result = subprocess.run(
        ["python", script_path, csv_path, "0.2", str(norm_val), "50"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8'
    )

    print(result.stdout)

    for line in result.stdout.splitlines():
        if "Returned accuracy:" in line:
            acc = float(line.split(":")[1].strip())
            accuracies.append(acc)
            norm_values.append(norm_val)
            print(f"Captured accuracy: {acc}")
            break

if accuracies:
    max_acc = max(accuracies)
    max_acc_norm = norm_values[accuracies.index(max_acc)]
    print(f"\nMax accuracy: {max_acc} at normalization value: {max_acc_norm}")

    plt.figure(figsize=(10, 6))
    plt.plot(norm_values, accuracies, marker='o', linestyle='-')
    plt.xlabel('Normalization Value (1–254)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Normalization Value')
    plt.grid(True)

    last_acc = None
    for x, y in zip(norm_values, accuracies):
        if last_acc is None or abs(y - last_acc) > 1e-4:
            plt.text(x, y + 0.001, f"{y:.3f}", ha='center', fontsize=6)
            last_acc = y

    plt.show()
else:
    print("No accuracies collected! Check script outputs.")
