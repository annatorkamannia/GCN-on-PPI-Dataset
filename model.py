import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

# Load the training, validation, and test datasets
train_dataset = PPI(root='/tmp/PPI', split='train')
val_dataset = PPI(root='/tmp/PPI', split='val')
test_dataset = PPI(root='/tmp/PPI', split='test')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(50, 128)  # Input: 50 (PPI features), Output: 64 hidden units
        self.conv2 = GCNConv(128, 128)  # Another GCN layer with 64 hidden units
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 121)  # Output: 121 classes (multi-label)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First graph convolution layer
        x = F.relu(x)
        x = self.conv2(x, edge_index)  # Second graph convolution layer
        x = F.relu(x)
        x = self.conv3(x, edge_index) # Third graph convolution layer
        x = F.relu(x)
        x = self.conv4(x, edge_index)  # Output layer
        return torch.sigmoid(x)  # Sigmoid for multi-label classification


# Move the model to the correct device (GPU or CPU)
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)

# Binary Cross Entropy loss for multi-label classification
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy(out, data.y.float())  # Multi-label loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = (out > 0.5).float()  # Apply threshold to classify labels
        correct += pred.eq(data.y).sum().item()
    return correct / (len(loader.dataset) * 121)  # 121 is the number of labels


# Train and validate the model for 50 epochs
for epoch in range(100):
    loss = train()
    val_acc = test(val_loader)
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}')


test_acc = test(test_loader)
print(f'Test Accuracy: {test_acc:.4f}')

# Save the model
torch.save(model.state_dict(), 'gcn_model_ppi.pth')

# To load the model
model = GCN()
model.load_state_dict(torch.load('gcn_model_ppi.pth'))
model.to(device)
