import subprocess
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings globally
warnings.filterwarnings("ignore", category=ConvergenceWarning)

accuracies = []
max_iters = []

# Range of max_iter values to test
max_iter_values = np.arange(1, 200, 10)

for max_iter in max_iter_values:
    print(f"Running: python tuned_mnist_logistic_regression_parameter.py 0.2 8.0 {max_iter} 42 lbfgs 1.0")

    result = subprocess.run(
    ["python.exe", ".\\tuned_mnist_logistic_regression_parameter.py",
     "0.2",          # test_size fixed
     "9.0",          # normalization_factor fixed
     str(max_iter),  # max_iter varies here
     "44",           # random_state fixed here
     "lbfgs",        # solver fixed
     "1.0"           # regularization C fixed
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
            max_iters.append(max_iter)
            print(f"Captured accuracy: {acc}")
            break

if accuracies:
    max_acc = max(accuracies)
    max_acc_iter = max_iters[accuracies.index(max_acc)]
    print(f"\nMax accuracy: {max_acc} at iteration: {max_acc_iter}")

    plt.figure(figsize=(10, 6))
    plt.plot(max_iters, accuracies, marker='o', linestyle='-')
    plt.xlabel('Max Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Max Iterations')
    plt.grid(True)

    last_acc = None
    for x, y in zip(max_iters, accuracies):
        if last_acc is None or abs(y - last_acc) > 1e-4:
            plt.text(x, y + 0.001, f"{y:.2f}", ha='center', fontsize=6)
            last_acc = y

    plt.show()
else:
    print("No accuracies collected! Check script outputs.")
