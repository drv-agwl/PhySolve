import pickle
import matplotlib.pyplot as plt

with open("./logs/LfM/100.pkl", "rb") as f:
    data = pickle.load(f)

train_losses = data["Train losses"]
val_losses = data["Test losses"]

print("Best train loss: ", min(train_losses))
print("Best val loss: ", min(val_losses), f"at epoch - {val_losses.index(min(val_losses))}")

epochs = range(1, len(train_losses)+1)

plt.plot(epochs, train_losses, label="train")

plt.plot(epochs, val_losses, label="val")

plt.legend()
plt.show()