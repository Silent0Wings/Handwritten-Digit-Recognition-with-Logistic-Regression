import subprocess
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']

accuracies_per_solver = {}

for solver in solvers:
    print(f"\n=== Testing solver: {solver} ===")
    
    result = subprocess.run(
    ["python", "tuned_mnist_logistic_regression_parameter.py",
     "0.2",        # test_size fixed
     "9.0",        # normalization_factor fixed
     "128",      # max_iter fixed
     "44",         # random_state fixed
     solver,       # solver varies here
     "1.0"         # regularization C fixed
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

    
    print(result.stdout)

    for line in result.stdout.splitlines():
        if "Returned accuracy:" in line:
            acc = float(line.split(":")[1].strip())
            accuracies_per_solver[solver] = acc
            print(f"Captured accuracy for {solver}: {acc:.5f}")
            break

# Plot results with distinct colors
plt.figure(figsize=(10, 6))

colors = [plt.cm.tab10(i) for i in range(len(accuracies_per_solver))]

plt.bar(
    accuracies_per_solver.keys(),
    accuracies_per_solver.values(),
    color=colors
)

plt.xlabel('Solver')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Across Solvers')
plt.ylim(0.963, 1)
plt.grid(axis='y')
plt.show()
