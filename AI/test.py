import matplotlib.pyplot as plt

# test.txt contains lines like this:
# epoch  47 | train loss 0.3344 tok-acc 0.857 | test loss 0.3408 tok-acc 0.854

train_loss = []
train_acc = []

test_loss = []
test_acc = []

with open('test.txt', 'r') as f:
    for line in f:
        parts = line.split('|')
        train_loss.append(float(parts[1].split()[2]))
        train_acc.append(float(parts[1].split()[4]))
        test_loss.append(float(parts[2].split()[2]))
        test_acc.append(float(parts[2].split()[4]))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
