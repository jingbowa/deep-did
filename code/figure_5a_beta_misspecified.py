import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import json
from typing import Callable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 12345

def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(tensor):
    return tensor.to(DEVICE)

def sigmoid_sum_treatment_prob(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.mean(x-5, axis=1)))

def multiplicative_fixed_effect(fe_i: np.ndarray, fe_t: np.ndarray) -> np.ndarray:
    return fe_i * fe_t

def linear_beta(x: np.ndarray) -> np.ndarray:
    return 15 * x[:, 0] - x[:, 1] * 2 + x[:, 2]

def linear_beta_all(x: np.ndarray) -> np.ndarray:
    weights = np.linspace(10, -10, x.shape[1])
    return np.mean(x * weights, axis=1)

def non_linear_beta(x: np.ndarray) -> np.ndarray:
    return 15 * x[:, 0] - (x[:, 1] ** 2) + np.sin(x[:, 2]) * 1

def simulate_panel_data(n_individuals: int, n_time_periods: int, x_dim: int, beta_func: Callable[[np.ndarray], np.ndarray]):
    cutoff_period = n_time_periods // 2
    
    X_i0 = np.random.normal(loc=5, scale=3, size=(n_individuals, x_dim))
    X = np.repeat(X_i0[:, np.newaxis, :], n_time_periods, axis=1)
        
    treatment_probs = sigmoid_sum_treatment_prob(X_i0)
    
    D = np.zeros((n_individuals, n_time_periods))
    treated_individuals = np.random.binomial(n=1, p=treatment_probs, size=n_individuals)
    D[:, cutoff_period:] = treated_individuals[:, np.newaxis]

    fe_i = np.random.normal(loc=0, scale=5, size=(n_individuals, 1))
    fe_t = np.random.normal(loc=0, scale=1, size=(1, n_time_periods))
    
    beta = beta_func(X_i0)
    beta_panel = np.repeat(beta[:, np.newaxis], n_time_periods, axis=1)
    
    epsi = np.random.normal(loc=0, scale=1, size=(n_individuals, n_time_periods))
    fe_interaction = multiplicative_fixed_effect(fe_i, fe_t)
    
    y = beta_panel * D + fe_interaction + epsi
    
    y = y.reshape(-1, 1, order='F')
    D = D.reshape(-1, 1, order='F')
    X = X.reshape(-1, x_dim, order='F')
    beta = beta.reshape(-1, 1, order='F')
    
    i_idx = np.repeat(np.arange(n_individuals), n_time_periods)
    t_idx = np.tile(np.arange(n_time_periods), n_individuals)
    
    return y, D, X, i_idx, t_idx, beta, X_i0

class PanelDataset(Dataset):
    def __init__(self, D: np.ndarray, X: np.ndarray, y: np.ndarray, i_idx: np.ndarray, t_idx: np.ndarray):
        self.D = to_device(torch.tensor(D, dtype=torch.float32))
        self.X = to_device(torch.tensor(X, dtype=torch.float32))
        self.y = to_device(torch.tensor(y, dtype=torch.float32))
        self.i_idx = to_device(torch.tensor(i_idx, dtype=torch.long))
        self.t_idx = to_device(torch.tensor(t_idx, dtype=torch.long))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.D[idx], self.X[idx], self.y[idx], self.i_idx[idx], self.t_idx[idx]

class LinearBetaLayer(nn.Module):
    def __init__(self, x_dim: int):
        super().__init__()
        self.linear = nn.Linear(x_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class NonLinearBetaLayer(nn.Module):
    def __init__(self, x_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(x_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class RegressionModel(nn.Module):
    def __init__(self, n_individuals: int, n_time_periods: int, x_dim: int, beta_layer: str, embedding_dim: int = 16):
        super().__init__()
        if beta_layer == 'Linear':
            self.beta_layers = LinearBetaLayer(x_dim)
        elif beta_layer == 'Nonlinear':
            self.beta_layers = NonLinearBetaLayer(x_dim)
        else:
            raise ValueError("beta_layer must be 'Linear' or 'Nonlinear'")

        self.i_embedding = nn.Embedding(n_individuals, embedding_dim)
        self.t_embedding = nn.Embedding(n_time_periods, embedding_dim)
        self.fe_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, d: torch.Tensor, x: torch.Tensor, i_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        beta = self.beta_layers(x)
        i_emb = self.i_embedding(i_idx)
        t_emb = self.t_embedding(t_idx)
        fe = self.fe_layers(torch.cat((i_emb, t_emb), 1))
        return beta * d + fe

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int, learning_rate: float = 1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for D, X, y, i_idx, t_idx in train_loader:
            optimizer.zero_grad()
            y_pred = model(D, X, i_idx, t_idx)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for D, X, y, i_idx, t_idx in val_loader:
                y_pred = model(D, X, i_idx, t_idx)
                val_loss += criterion(y_pred, y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch: {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

def run_simulation(n_individuals: int, n_time_periods: int, x_dim: int, beta_func: Callable[[np.ndarray], np.ndarray], beta_layer: str):
    y_train, D_train, X_train, i_idx_train, t_idx_train, _, X_i0_train = simulate_panel_data(n_individuals, n_time_periods, x_dim, beta_func)
    
    dataset = PanelDataset(D_train, X_train, y_train, i_idx_train, t_idx_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
    
    model = RegressionModel(n_individuals, n_time_periods, x_dim, beta_layer).to(DEVICE)
    train_model(model, train_loader, val_loader, n_epochs=10)
    
    y_test, D_test, X_test, i_idx_test, t_idx_test, beta_test, X_i0_test = simulate_panel_data(n_individuals, n_time_periods, x_dim, beta_func)

    true_beta = beta_test.flatten()
    
    model.eval()
    with torch.no_grad():
        estimated_beta = model.beta_layers(to_device(torch.tensor(X_i0_test, dtype=torch.float32))).cpu().numpy().flatten()

    return true_beta, estimated_beta

def save_simulation_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f)

def main():
    n_individuals = 3000
    n_time_periods = 90
    x_dim = 100

    results = {}
    for true_beta_func, true_beta_name in [(linear_beta, 'Linear'), (linear_beta_all, 'LinearAll'), (non_linear_beta, 'Nonlinear')]:
        for beta_layer in ['Linear', 'Nonlinear']:
            print(f"Running simulation for true beta: {true_beta_name}, beta layer: {beta_layer}")
            set_seed()
            true_beta, estimated_beta = run_simulation(n_individuals, n_time_periods, x_dim, true_beta_func, beta_layer)
            
            key = f"{true_beta_name}_{beta_layer}"
            results[key] = {
                'true_beta': true_beta.tolist(),
                'estimated_beta': estimated_beta.tolist(),
            }

    save_simulation_results(results, './simulated_data/data_5a_beta_misspecified.txt')

if __name__ == "__main__":
    main()
