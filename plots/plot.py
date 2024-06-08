import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pal = sns.mpl_palette("viridis")

pal = [pal[0], pal[3], pal[5]]


def make_ema(array, alpha=0.6):
    arr = [array[0]]
    for i in range(1, len(array)):
        arr.append(
            alpha * array[i] + (1 - alpha) * arr[i - 1]
        )  # arr[i-1] because we need the last
    return arr


noise = 0.2

fig = plt.figure(figsize=(7, 5))
data = pd.read_csv("plots/data.csv")
data = data[(data["Loss"] == "Validation") & (data["Noise"] == noise)]
for v in ["v0", "v1", "v2"]:
    arr = list(data[data["Version"] == v]["Value"])
    arr2 = make_ema(arr, alpha=0.35)
    data.loc[data["Version"] == v, "Value"] = arr2

sns.lineplot(data=data, x="Step", y="Value", hue="Version", palette=pal)
plt.yscale("log")
plt.ylabel("Validation Loss")
plt.title("Validation loss (smoothed)")
plt.show()
