import subprocess
import matplotlib.pyplot as plt

accuracies = []
test_sizes = []

for i in range(1, 10):
    test_size = round(i * 0.1, 1)
    print(f"Running: python tuned_mnist_logistic_regression_parameter.py {test_size} 8.0 10000 42 lbfgs 1.0")

    result = subprocess.run(
    ["python", "tuned_mnist_logistic_regression_parameter.py",
     str(test_size),    # test_size varies here
     "9.0",             # normalization_factor fixed here
     "128",             # max_iter fixed here
     "44",              # random_state fixed here
     "lbfgs",           # solver fixed here
     "1.0"              # regularization C fixed here
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
            test_sizes.append(test_size)
            print(f"Captured accuracy: {acc}")
            break

if accuracies:
    plt.figure(figsize=(8, 5))
    plt.plot(test_sizes, accuracies, marker='o', linestyle='-')
    plt.xlabel('Test Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Test Size')
    plt.grid(True)

    for x, y in zip(test_sizes, accuracies):
        plt.text(x, y + 0.001, f"{y:.3f}", ha='center', fontsize=8)

    plt.show()
else:
    print("No accuracies collected! Check script outputs.")
