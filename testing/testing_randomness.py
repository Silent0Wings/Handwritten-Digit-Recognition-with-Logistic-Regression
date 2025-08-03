import subprocess 
import matplotlib.pyplot as plt

accuracies = []
random_seeds = []

for random_state in range(1, 50):
    print(f"Running: python tuned_mnist_logistic_regression_parameter.py 0.2 8.0 50 {random_state} lbfgs")
    
    result = subprocess.run(
    ["python", "tuned_mnist_logistic_regression_parameter.py",
     "0.2",        # test_size fixed
     "9.0",        # normalization_factor fixed
     "128",       # max_iter fixed
     str(random_state),  # random_state varies here
     "lbfgs"       # solver fixed
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
            random_seeds.append(random_state)
            print(f"Captured accuracy: {acc}")
            break

if accuracies:
    plt.figure(figsize=(8, 5))
    plt.plot(random_seeds, accuracies, marker='o', linestyle='-')
    plt.xlabel('Random Seed')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Random Seed')
    plt.grid(True)

    for x, y in zip(random_seeds, accuracies):
        plt.text(x, y + 0.001, f"{y:.3f}", ha='center', fontsize=8)

    plt.show()
else:
    print("No accuracies collected! Check script outputs.")
