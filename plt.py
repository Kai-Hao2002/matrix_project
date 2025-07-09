import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('benchmark_results_multi.csv')

methods = df['method'].unique()
n_values = sorted(df['n'].unique())

plt.figure(figsize=(10,6))

for method in methods:
    sub_df = df[df['method'] == method]
    sub_df = sub_df.set_index('n').loc[n_values]
    plt.plot(n_values, sub_df['gflops'], marker='o', label=method)

plt.xlabel('Matrix size (n)')
plt.ylabel('GFLOPS')
plt.title('Performance Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('benchmark_by_size.png')
plt.show()
