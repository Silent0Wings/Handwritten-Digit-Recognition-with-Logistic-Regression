import subprocess
import matplotlib.pyplot as plt

accuracies = []
c_values = []

c_list = [0.001, 0.01, 0.05, 0.1, 0.21, 0.5, 1, 2, 5, 10, 50, 100]

for c in c_list:
    print(f"Running: python tuned_mnist_logistic_regression_parameter.py 0.2 8.0 50 42 lbfgs {c}")

    result = subprocess.run(
    ["python.exe", ".\\tuned_mnist_logistic_regression_parameter.py",
     "0.2",       # test_size fixed
     "9.0",       # normalization_factor fixed
     "128",       # max_iter fixed
     "44",        # random_state fixed
     "lbfgs",     # solver fixed
     str(c)       # regularization C passed as variable
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
            c_values.append(c)
            print(f"Captured accuracy: {acc}")
            break

if accuracies:
    plt.figure(figsize=(8, 5))
    plt.plot(c_values, accuracies, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Regularization Strength (C)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Regularization (C)')
    plt.grid(True)

    for x, y in zip(c_values, accuracies):
        plt.text(x, y + 0.001, f"{y:.3f}", ha='center', fontsize=8)

    plt.show()
else:
    print("No accuracies collected! Check script outputs.")
