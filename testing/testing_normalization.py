import subprocess
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings globally
warnings.filterwarnings("ignore", category=ConvergenceWarning)

accuracies = []
norm_values = []

# Test normalization values from 1 to 16 inclusive
norm_test_values = np.arange(1, 17, 1)

for norm_val in norm_test_values:
    print(f"Running: python tuned_mnist_logistic_regression_parameter.py 0.2 {norm_val} 128 44 lbfgs 1.0")

    result = subprocess.run(
        ["python.exe", ".\\tuned_mnist_logistic_regression_parameter.py",
         "0.2",
         str(norm_val),
         "128",
         "44",
         "lbfgs",
         "1.0"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
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
    plt.xlabel('Normalization Value (1–16)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Normalization Value')
    plt.ylim(0.98, 1)  # <—— LIMIT y-axis between 0.98 and 1
    plt.grid(True)

    last_acc = None
    for x, y in zip(norm_values, accuracies):
        if last_acc is None or abs(y - last_acc) > 1e-4:
            plt.text(x, y + 0.001, f"{y:.4f}", ha='center', fontsize=6)
            last_acc = y

    plt.show()
else:
    print("No accuracies collected! Check script outputs.")
