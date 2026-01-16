import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


csv_files = glob.glob('./runtime_*.csv')
files = []
pattern = re.compile(r'runtime_.*_(\d+)_n(\d+)\.csv')

for path in csv_files:
    base = os.path.basename(path)
    m = pattern.match(base)
    if m:
        sample = int(m.group(1))
        feat = int(m.group(2))
        files.append((path, sample, feat))

# 先按 sample，再按 feat 排序
files.sort(key=lambda x: (x[1], x[2]))
files
'''
files =[
 #('./runtime_sample20_feat10.csv', 20, 5),
 ('./runtime_sample20_feat10.csv', 20, 10),
 ('./runtime_sample20_feat15.csv', 20, 15),
 #('./runtime_sample200_feat10.csv', 200, 5),
 ('./runtime_sample200_feat10.csv', 200, 10),
 ('./runtime_sample200_feat15.csv', 200, 15)]
'''


plt.figure(figsize=(7, 5))

# === 映射规则 ===
sample_to_ls = {
    20: '-',
    200: '--'
}

feat_to_color = {
    10: 'tab:blue',
    15: 'tab:orange'
}

# === 画线 ===
for path, sample, feat in files:
    if sample not in sample_to_ls or feat not in feat_to_color:
        continue  # 只画 sample={20,200}, feat={10,15}

    df = pd.read_csv(path)

    plt.plot(
        df['degree'],
        df['runtime_seconds'],
        linestyle=sample_to_ls[sample],
        color=feat_to_color[feat],
        alpha=0.85
    )

# === 自定义 legend（关键点） ===
legend_elements = [
    # sample legend
    Line2D([0], [0], color='black', linestyle='-', label='sample=20'),
    Line2D([0], [0], color='black', linestyle='--', label='sample=200'),

    # feature legend
    Line2D([0], [0], color='tab:blue', linestyle='-', label='feat=10'),
    Line2D([0], [0], color='tab:orange', linestyle='-', label='feat=15'),
]

plt.legend(
    handles=legend_elements,
    fontsize=9,
    ncol=1,
    frameon=True
)
plt.yscale('log')
plt.xlabel("Max degree")
plt.ylabel("Runtime (log(seconds))")
plt.title("Runtime vs Degree")
plt.grid(True)
plt.tight_layout()
plt.savefig('./edge_limits_plot_encoded.png', dpi=300)
plt.show()
