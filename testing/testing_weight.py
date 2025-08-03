import subprocess
import matplotlib.pyplot as plt

accuracies = []
weights = []

weight_options = ['None', 'balanced']
colors = ['skyblue', 'orange']  # Different colors for bars

for weight in weight_options:
    print(f"Running: python tuned_mnist_logistic_regression_parameter.py 0.2 8.0 50 42 lbfgs 1.0 {weight}")

    result = subprocess.run(
    ["python", "tuned_mnist_logistic_regression_parameter.py",
     "0.2",        # test_size fixed
     "9.0",        # normalization_factor fixed
     "128",         # max_iter fixed
     "44",         # random_state fixed
     "lbfgs",      # solver fixed
     "1.0",        # regularization C fixed
     weight        # class_weight varies here
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
            weights.append(weight)
            print(f"Captured accuracy: {acc}")
            break

if accuracies:
    plt.figure(figsize=(8, 5))
    plt.bar(weights, accuracies, color=colors)
    plt.xlabel('Class Weight Setting')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Class Weight')
    plt.grid(True, axis='y')

    for x, y in zip(weights, accuracies):
        plt.text(x, y + 0.001, f"{y:.3f}", ha='center', fontsize=8)

    plt.show()
else:
    print("No accuracies collected! Check script outputs.")
