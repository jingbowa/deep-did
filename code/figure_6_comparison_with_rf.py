import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def true_function(X):
    return (
        5 * np.sin(X[:, 0]) + 
        2 * np.cos(X[:, 1] * X[:, 2]) + 
        3 * np.exp(-((X[:, 3] - 2)**2 + (X[:, 4] + 1)**2) / 10) +
        np.sum(X[:, 5:], axis=1) #linear all other features
    )

def generate_data(n_samples, n_features=100):
    X = np.random.uniform(-10, 10, (n_samples, n_features))
    y = true_function(X) + np.random.normal(0, 1, n_samples)
    return X, y.reshape(-1, 1) # Ensure y is a column vector

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

def train_pytorch_model(model, train_loader, n_epochs=20, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

def save_results(X_test, y_test, y_pred_nn, y_pred_rf):
    results = {
        'X_test': X_test.tolist(),
        'y_test': y_test.flatten().tolist(),
        'y_pred_nn': y_pred_nn.flatten().tolist(),
        'y_pred_rf': y_pred_rf.flatten().tolist()
    }
    
    with open('./simulated_data/data_6_comparison_with_rf.txt', 'w') as f:
        json.dump(results, f)

def main():
    n_features = 100
    X, y = generate_data(50_000, n_features=n_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    nn_model_pytorch = MLP(input_dim=n_features).to(DEVICE)
    train_pytorch_model(nn_model_pytorch, train_loader, n_epochs=20)

    nn_model_pytorch.eval()
    with torch.no_grad():
        y_pred_nn_tensor = nn_model_pytorch(X_test_tensor.to(DEVICE))
    y_pred_nn = y_pred_nn_tensor.cpu().numpy()

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train.ravel())
    y_pred_rf = rf_model.predict(X_test)

    mse_nn = mean_squared_error(y_test, y_pred_nn)
    mse_rf = mean_squared_error(y_test, y_pred_rf)

    print("Mean Squared Error:")
    print(f"Neural Network: {mse_nn:.4f}")
    print(f"Random Forest: {mse_rf:.4f}")

    save_results(X_test, y_test, y_pred_nn, y_pred_rf)

if __name__ == "__main__":
    main()
