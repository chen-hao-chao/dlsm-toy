from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})

log_dir = "./results"
df = SummaryReader(log_dir, cols={'dir_name'}).scalars
df = df[["step", "loss_ce", "total Dl", "dir_name"]]
df = df.assign(dir_name=df["dir_name"].apply(lambda s: s.split('/')[0]))

fig, ax = plt.subplots(2, 1, figsize=(8.4,6), dpi=80, linewidth = 1)
plt.sca(ax[0])
sns.lineplot(data=df, x='step', y='total Dl', hue='dir_name')
plt.legend([])

plt.sca(ax[1])
sns.lineplot(data=df, x='step', y='loss_ce', hue='dir_name')
plt.legend([])
plt.tight_layout()
plt.savefig('example.eps')