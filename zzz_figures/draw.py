import matplotlib.pyplot as plt
import pandas as pd

# Define the columns and rows of interest
# columns_of_interest = ['Pretrain 2000 epochs', 'Pretrain 1000 epochs', 'Pretrain 200 epochs', 'IQN-filter_ratio0.05_iqn1.5', 'IQN-filter_ratio0.02_iqn1.5', 'IQN-filter_ratio0.008_iqn1.5', "Z-score_thres30", "Z-score_ratio0.01_thres3"]
columns_of_interest = ['Pretrain 2000 epochs', 'Pretrain 1000 epochs', 'Pretrain 200 epochs', 'IQN-filter_ratio0.05_iqn1.5', 'IQN-filter_ratio0.02_iqn1.5', 'IQN-filter_ratio0.008_iqn1.5', "Z-score_thres30"]
rows_of_interest = ['ETTh1 + pl96', 'ETTh1 + pl720', 'ETTh2 + pl96', 'ETTm1 + pl96', 'ETTm2 + pl96', 'ECL + pl96', 'Weather + pl96']

# Assuming the data is in a DataFrame named `df`, we'll create a sample DataFrame to mimic the structure.
# In practice, you would replace this with `df = pd.read_excel(file_path, sheet_name='Sheet1')` or the appropriate sheet.

# Sample DataFrame structure
data = {
    'Metric': rows_of_interest,
    
    'Pretrain 2000 epochs': [0.373, 0.462, 0.28, 0.355, 0.214, 0.191, 0.189],
    'Pretrain 1000 epochs': [0.382, 0.447, 0.281, 0.381, 0.208, 0.204, 0.188],
    'Pretrain 200 epochs': [0.39, 0.45, 0.281, 0.47, 0.228, 0.244, 0.212],
    'IQN-filter_ratio0.05_iqn1.5': [0.404, 0.467, 0.285, 0.396, 0.222, 0.233, 0.202],
    'IQN-filter_ratio0.02_iqn1.5': [0.428, 0.497, 0.308, 0.411, 0.235, 0.228, 0.21],
    'IQN-filter_ratio0.008_iqn1.5': [0.441, 0.504, 0.31, 0.395, 0.229, 0.21, 0.206],
    'Z-score_thres30':[0.404, 0.472, 0.292, 0.443, 0.225, 0.241, 0.201],
    # 'Z-score_ratio0.01_thres3': [0.735, 0.818, 0.347, 0.759, 0.245, 0.858, 0.319],
}
df = pd.DataFrame(data)

# Plotting the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Set bar width
bar_width = 0.1

# Set positions of the bars on the x-axis
r1 = range(len(df))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]
r6 = [x + bar_width for x in r5]
r7 = [x + bar_width for x in r6]
# r8 = [x + bar_width for x in r7]

# Create bars
ALPHA = 0.7
ax.bar(r1, df['Pretrain 2000 epochs'], color='blue', width=bar_width, edgecolor='grey', label='Pretrain 2000 epochs', alpha=ALPHA)
ax.bar(r2, df['Pretrain 1000 epochs'], color='green', width=bar_width, edgecolor='grey', label='Pretrain 1000 epochs', alpha=ALPHA)
ax.bar(r3, df['Pretrain 200 epochs'], color='red', width=bar_width, edgecolor='grey', label='Pretrain 200 epochs', alpha=ALPHA)
ax.bar(r4, df['IQN-filter_ratio0.05_iqn1.5'], color='cyan', width=bar_width, edgecolor='grey', label='IQN-filter_ratio0.05_iqn1.5', alpha=ALPHA)
ax.bar(r5, df['IQN-filter_ratio0.02_iqn1.5'], color='magenta', width=bar_width, edgecolor='grey', label='IQN-filter_ratio0.02_iqn1.5', alpha=ALPHA)
ax.bar(r6, df['IQN-filter_ratio0.008_iqn1.5'], color='yellow', width=bar_width, edgecolor='grey', label='IQN-filter_ratio0.008_iqn1.5', alpha=ALPHA)
ax.bar(r7, df["Z-score_thres30"], color='purple', width=bar_width, edgecolor='grey', label='Z-score_thres30', alpha=ALPHA)
# ax.bar(r8, df["Z-score_ratio0.01_thres3"], color='orange', width=bar_width, edgecolor='grey', label='Z-score_ratio0.01_thres3', alpha=ALPHA)


# Add xticks on the middle of the group bars
ax.set_xlabel('Metrics', fontweight='bold')
ax.set_ylabel('Values', fontweight='bold')
ax.set_xticks([r + bar_width for r in range(len(df))])
ax.set_xticklabels(df['Metric'], rotation=45, ha="right")

# Create legend & Show graphic
ax.legend()
plt.tight_layout()

plt.show()
plt.savefig("comparison.pdf")