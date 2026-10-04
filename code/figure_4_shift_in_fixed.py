import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 12345

def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(tensor):
    return tensor.to(DEVICE)

def create_onehot_encodings(i_idx: np.ndarray, t_idx: np.ndarray, n_individuals: int, n_time_periods: int):
    i_onehot = np.eye(n_individuals)[i_idx]
    t_onehot = np.eye(n_time_periods)[t_idx]
    return i_onehot, t_onehot

def simulate_panel_data(n_individuals: int, n_time_periods: int, x_dim: int, scenario: int):
    cutoff_period = n_time_periods // 2
    
    high_low_group_assignment = np.random.binomial(n=1, p=0.5, size=n_individuals)
    X_i0 = np.where(high_low_group_assignment[:, np.newaxis] == 1,
                    np.random.normal(loc=8, scale=3, size=(n_individuals, x_dim)),
                    np.random.normal(loc=5, scale=3, size=(n_individuals, x_dim)))
    X = np.repeat(X_i0[:, np.newaxis, :], n_time_periods, axis=1)
    
    if scenario == 1:
        fe_i = np.where(high_low_group_assignment[:, np.newaxis] == 1,
                        np.random.normal(loc=2, scale=5, size=(n_individuals, 1)),
                        np.random.normal(loc=0, scale=5, size=(n_individuals, 1)))
        fe_t = np.random.normal(loc=0, scale=1, size=(1, n_time_periods))
    elif scenario == 2:
        fe_i = np.random.normal(loc=0, scale=5, size=(n_individuals, 1))
        fe_t_pre = np.random.normal(loc=0, scale=1, size=(1, cutoff_period))
        fe_t_post = np.random.normal(loc=2, scale=1, size=(1, n_time_periods - cutoff_period))
        fe_t = np.concatenate([fe_t_pre, fe_t_post], axis=1)
    else:  # scenario 3
        fe_i = np.where(high_low_group_assignment[:, np.newaxis] == 1,
                        np.random.normal(loc=2, scale=5, size=(n_individuals, 1)),
                        np.random.normal(loc=0, scale=5, size=(n_individuals, 1)))
        fe_t_pre = np.random.normal(loc=0, scale=1, size=(1, cutoff_period))
        fe_t_post = np.random.normal(loc=2, scale=1, size=(1, n_time_periods - cutoff_period))
        fe_t = np.concatenate([fe_t_pre, fe_t_post], axis=1)
    
    selection_index = np.sum(X_i0[:, :3], axis=1) + fe_i.flatten()
    selection_index = (selection_index - selection_index.min()) / (selection_index.max() - selection_index.min())
    treatment_probs = 1 / (1 + np.exp(-selection_index))
    treated_individuals = np.random.binomial(n=1, p=treatment_probs, size=n_individuals)
    D = np.zeros((n_individuals, n_time_periods))
    D[:, cutoff_period:] = treated_individuals[:, np.newaxis]
    
    beta = 15 * X_i0[:, 0] - X_i0[:, 1] * 2 + X_i0[:, 2]
    beta_panel = np.repeat(beta[:, np.newaxis], n_time_periods, axis=1)
    
    epsi = np.random.normal(loc=0, scale=1, size=(n_individuals, n_time_periods))
    fe_interaction = fe_i + fe_t
    y = beta_panel * D + fe_interaction + epsi
    
    i_idx = np.repeat(np.arange(n_individuals), n_time_periods)
    t_idx = np.tile(np.arange(n_time_periods), n_individuals)
    
    y = y.reshape(-1, 1, order='F')
    D = D.reshape(-1, 1, order='F')
    X = X.reshape(-1, x_dim, order='F')
    beta = beta.reshape(-1, 1, order='F')
    
    i_onehot, t_onehot = create_onehot_encodings(i_idx, t_idx, n_individuals, n_time_periods)
    
    return y, D, X, i_onehot, t_onehot, beta, X_i0

class PanelDataset(Dataset):
    def __init__(self, D: np.ndarray, X: np.ndarray, y: np.ndarray, i_onehot: np.ndarray, t_onehot: np.ndarray):
        self.D = to_device(torch.tensor(D, dtype=torch.float32))
        self.X = to_device(torch.tensor(X, dtype=torch.float32))
        self.y = to_device(torch.tensor(y, dtype=torch.float32))
        self.i_onehot = to_device(torch.tensor(i_onehot, dtype=torch.float32))
        self.t_onehot = to_device(torch.tensor(t_onehot, dtype=torch.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.D[idx], self.X[idx], self.y[idx], self.i_onehot[idx], self.t_onehot[idx]

class RegressionModel(nn.Module):
    def __init__(self, n_individuals: int, n_time_periods: int, x_dim: int):
        super().__init__()
        self.beta_layers = nn.Sequential(
            nn.Linear(x_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.fe_layers = nn.Sequential(
            nn.Linear(n_individuals + n_time_periods, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, d: torch.Tensor, x: torch.Tensor, i_onehot: torch.Tensor, t_onehot: torch.Tensor) -> torch.Tensor:
        beta = self.beta_layers(x)
        fe = self.fe_layers(torch.cat((i_onehot, t_onehot), 1))
        return beta * d + fe

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int, learning_rate: float = 0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for D, X, y, i_onehot, t_onehot in train_loader:
            optimizer.zero_grad()
            y_pred = model(D, X, i_onehot, t_onehot)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for D, X, y, i_onehot, t_onehot in val_loader:
                y_pred = model(D, X, i_onehot, t_onehot)
                val_loss += criterion(y_pred, y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch: {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

def run_simulation(n_individuals: int, n_time_periods: int, x_dim: int, scenario: int):
    y_train, D_train, X_train, i_onehot_train, t_onehot_train, _, X_i0_train = simulate_panel_data(n_individuals, n_time_periods, x_dim, scenario)
    
    dataset = PanelDataset(D_train, X_train, y_train, i_onehot_train, t_onehot_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
    
    model = RegressionModel(n_individuals, n_time_periods, x_dim).to(DEVICE)
    train_model(model, train_loader, val_loader, n_epochs=10)
    
    y_test, D_test, X_test, i_onehot_test, t_onehot_test, beta_test, X_i0_test = simulate_panel_data(n_individuals, n_time_periods, x_dim, scenario)

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
    
    results = {
        'n_individuals': n_individuals,
        'n_time_periods': n_time_periods,
        'x_dim': x_dim,
        'scenarios': ['Individual FE', 'Time FE', 'Both FE'],
        'results': {}
    }
    
    for scenario in [1, 2, 3]:
        print(f"Running simulation for scenario {scenario}")
        set_seed()
        true_beta, estimated_beta = run_simulation(n_individuals, n_time_periods, x_dim, scenario)
        results['results'][f'scenario_{scenario}'] = {
            'true_beta': true_beta.tolist(),
            'estimated_beta': estimated_beta.tolist()
        }
    
    save_simulation_results(results, './simulated_data/data_4_shift_in_fixed.txt')

if __name__ == "__main__":
    main()
