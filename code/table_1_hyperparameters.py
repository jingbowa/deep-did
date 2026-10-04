import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import stats
from scipy.spatial.distance import jensenshannon
import pandas as pd
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 123456

def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(tensor):
    return tensor.to(DEVICE)

def sigmoid_sum_treatment_prob(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.mean(x-5, axis=1)))

def additive_fixed_effect(fe_i: np.ndarray, fe_t: np.ndarray) -> np.ndarray:
    return fe_i + fe_t

def linear_beta(x: np.ndarray) -> np.ndarray:
    return 15 * x[:, 0] - x[:, 1] * 2 + x[:, 2]

def simulate_panel_data(n_individuals: int, n_time_periods: int, x_dim: int):
    cutoff_period = n_time_periods // 2
    
    X_i0 = np.random.normal(loc=5, scale=3, size=(n_individuals, x_dim))
    X = np.repeat(X_i0[:, np.newaxis, :], n_time_periods, axis=1)
        
    treatment_probs = sigmoid_sum_treatment_prob(X_i0)
    
    D = np.zeros((n_individuals, n_time_periods))
    treated_individuals = np.random.binomial(n=1, p=treatment_probs, size=n_individuals)
    D[:, cutoff_period:] = treated_individuals[:, np.newaxis]

    fe_i = np.random.normal(loc=0, scale=5, size=(n_individuals, 1))
    fe_t = np.random.normal(loc=0, scale=1, size=(1, n_time_periods))
    
    beta = linear_beta(X_i0)
    beta_panel = np.repeat(beta[:, np.newaxis], n_time_periods, axis=1)
    
    epsi = np.random.normal(loc=0, scale=1, size=(n_individuals, n_time_periods))
    fe_interaction = additive_fixed_effect(fe_i, fe_t)
    
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

class RegressionModel(nn.Module):
    def __init__(self, n_individuals: int, n_time_periods: int, x_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.beta_layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.i_embedding = nn.Embedding(n_individuals, hidden_dim)
        self.t_embedding = nn.Embedding(n_time_periods, hidden_dim)
        self.fe_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, d: torch.Tensor, x: torch.Tensor, i_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        beta = self.beta_layers(x)
        i_emb = self.i_embedding(i_idx)
        t_emb = self.t_embedding(t_idx)
        fe = self.fe_layers(torch.cat((i_emb, t_emb), 1))
        return beta * d + fe

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int, optimizer: torch.optim.Optimizer):
    criterion = nn.MSELoss()
    
    for epoch in range(n_epochs):
        model.train()
        for D, X, y, i_idx, t_idx in train_loader:
            optimizer.zero_grad()
            y_pred = model(D, X, i_idx, t_idx)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for D, X, y, i_idx, t_idx in val_loader:
            y_pred = model(D, X, i_idx, t_idx)
            val_loss += criterion(y_pred, y).item()
    
    val_loss /= len(val_loader)
    return val_loss

def evaluate_performance(true_beta: np.ndarray, estimated_beta: np.ndarray):
    true_beta = np.asarray(true_beta)
    estimated_beta = np.asarray(estimated_beta)

    valid_mask = np.isfinite(estimated_beta) & np.isfinite(true_beta)
    true_beta_valid = true_beta[valid_mask]
    estimated_beta_valid = estimated_beta[valid_mask]

    if true_beta_valid.size == 0 or estimated_beta_valid.size == 0:
        return 1.0, 1.0, 1.0

    try:
        ks_statistic, _ = stats.ks_2samp(true_beta_valid, estimated_beta_valid)
    except ValueError:
        ks_statistic = 1.0

    if np.isnan(ks_statistic):
        ks_statistic = 1.0

    try:
        hist_true, bin_edges = np.histogram(true_beta_valid, bins=100, density=True)
        hist_est, _ = np.histogram(estimated_beta_valid, bins=bin_edges, density=True)
    except ValueError:
        return ks_statistic, 1.0, 1.0

    hist_true = hist_true + 1e-10
    hist_est = hist_est + 1e-10

    sum_true = np.sum(hist_true)
    sum_est = np.sum(hist_est)
    if sum_true == 0 or sum_est == 0:
        js_divergence = 1.0
    else:
        hist_true = hist_true / sum_true
        hist_est = hist_est / sum_est
        js_divergence = jensenshannon(hist_true, hist_est)

    if np.isnan(js_divergence):
        js_divergence = 1.0

    if true_beta_valid.size == 0:
        return ks_statistic, js_divergence, 1.0

    mse = np.mean((true_beta_valid - estimated_beta_valid)**2)
    variance_true_beta = np.var(true_beta_valid)
    nrmse = np.sqrt(mse / (variance_true_beta + 1e-10))

    if np.isnan(nrmse):
        nrmse = 1.0

    return ks_statistic, js_divergence, nrmse

def run_simulation(n_individuals: int, n_time_periods: int, x_dim: int, hidden_dim: int, batch_size: int, learning_rate: float, optimizer_name: str, n_epochs: int):
    y_train, D_train, X_train, i_idx_train, t_idx_train, _, X_i0_train = simulate_panel_data(n_individuals, n_time_periods, x_dim)
    
    dataset = PanelDataset(D_train, X_train, y_train, i_idx_train, t_idx_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = RegressionModel(n_individuals, n_time_periods, x_dim, hidden_dim=hidden_dim).to(DEVICE)
    
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    val_loss = train_model(model, train_loader, val_loader, n_epochs, optimizer)
    
    y_test, D_test, X_test, i_idx_test, t_idx_test, beta_test, X_i0_test = simulate_panel_data(n_individuals, n_time_periods, x_dim)

    true_beta = beta_test.flatten()
    
    model.eval()
    with torch.no_grad():
        estimated_beta = model.beta_layers(to_device(torch.tensor(X_i0_test, dtype=torch.float32))).cpu().numpy().flatten()

    ks_statistic, js_divergence, nrmse = evaluate_performance(true_beta, estimated_beta)
    
    return val_loss, ks_statistic, js_divergence, nrmse

def run_multiple_simulations(n_runs: int, **kwargs):
    results = []
    for _ in tqdm(range(n_runs), desc="Running simulations", leave=False):
        val_loss, ks_statistic, js_divergence, nrmse = run_simulation(**kwargs)
        results.append((val_loss, ks_statistic, js_divergence, nrmse))
    
    mean_results = {
        'val_loss': np.mean([r[0] for r in results]),
        'ks_statistic': np.mean([r[1] for r in results]),
        'js_divergence': np.mean([r[2] for r in results]),
        'nrmse': np.mean([r[3] for r in results])
    }
    
    return mean_results

def save_results_to_csv(results, filename="./table/table_hyperparameters.csv"):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    return df

def generate_latex_tables(df):
    latex_table = """\\begin{table}[h!tbp]
\\caption{Estimation Performance of Deep-DiD Across Different Implementation Choices}\\label{tab:sim_hyperparam}
\\scriptsize
{\\def\\arraystretch{1.2}  
\\begin{tabular}{cccccccc}
\\hline\\hline
\\multirow{2}{*}{\\centering \\#} & \\multirow{2}{*}{\\centering Optimizer} & \\# Nodes in Each & \\multirow{2}{*}{\\centering Batch Size} & Learning & \\multirow{2}{*}{\\centering Epochs} &  \\multirow{2}{*}{\\centering KS Statistic} & \\multirow{2}{*}{\\centering JS Divergence}          \\\\
&   & Hidden Layer     &                             & Rate     &                               &                               &  \\\\
\\hline
"""
    
    for i, row in df.iterrows():
        latex_table += f"{i+1}  & {row['Optimizer'].capitalize()}      & {row['Hidden Dim']}         & {row['Batch Size']}        & {row['Learning Rate']}         & {row['Epochs']}     & {row['KS Statistic']:.3f}        & {row['JS Divergence']:.3f}         \\\\\n"
    
    latex_table += """\\hline\\hline
\\end{tabular}
}
\\parbox{\\linewidth}{
\\vspace{1eX} 
\\scriptsize Note. The true functional form of $\\beta(\\mathbf{W}_{i})$ is ($15W_{i1} - 2W_{i2} + W_{i3}$). The true functional form of $f(\\mu_i, \\tau_t)$ is $\\mu_i + \\tau_t$. We set $N=3000, T=90, J=100$. SGD stands for stochastic gradient descent.}
\\end{table}"""

    latex_table_nrmse = """\\begin{table}[h!tbp]
\\caption{Estimation Performance of Deep-DiD Across Different Implementation Choices (with NRMSE)}\\label{tab:sim_hyperparam_nrmse}
\\scriptsize
{\\def\\arraystretch{1.2}  
\\begin{tabular}{ccccccccc}
\\hline\\hline
\\multirow{2}{*}{\\centering \\#} & \\multirow{2}{*}{\\centering Optimizer} & \\# Nodes in Each & \\multirow{2}{*}{\\centering Batch Size} & Learning & \\multirow{2}{*}{\\centering Epochs} &  \\multirow{2}{*}{\\centering KS Statistic} & \\multirow{2}{*}{\\centering JS Divergence} & \\multirow{2}{*}{\\centering NRMSE}          \\\\
&   & Hidden Layer     &                             & Rate     &                               &                               &  &  \\\\
\\hline
"""
    
    for i, row in df.iterrows():
        latex_table_nrmse += f"{i+1}  & {row['Optimizer'].capitalize()}      & {row['Hidden Dim']}         & {row['Batch Size']}        & {row['Learning Rate']}         & {row['Epochs']}     & {row['KS Statistic']:.3f}        & {row['JS Divergence']:.3f}         & {row['NRMSE']:.3f}         \\\\\n"
    
    latex_table_nrmse += """\\hline\\hline
\\end{tabular}
}
\\parbox{\\linewidth}{
\\vspace{1eX} 
\\scriptsize Note. The true functional form of $\\beta(\\mathbf{W}_{i})$ is ($15W_{i1} - 2W_{i2} + W_{i3}$). The true functional form of $f(\\mu_i, \\tau_t)$ is $\\mu_i + \\tau_t$. We set $N=3000, T=90, J=100$. SGD stands for stochastic gradient descent.}
\\end{table}"""
    
    return latex_table, latex_table_nrmse

def save_latex_tables(latex_table, latex_table_nrmse):
    with open('./table/table_hyperparameters.tex', 'w') as f:
        f.write(latex_table)
    
    with open('./table/table_hyperparameters_nrmse.tex', 'w') as f:
        f.write(latex_table_nrmse)

def main():
    n_individuals = 3000
    n_time_periods = 90
    x_dim = 100
    
    parameter_combinations = [
        {'hidden_dim': 32, 'batch_size': 512, 'lr': 0.001, 'optimizer': 'adam', 'epochs': 10},
        {'hidden_dim': 64, 'batch_size': 512, 'lr': 0.001, 'optimizer': 'adam', 'epochs': 10},
        {'hidden_dim': 32, 'batch_size': 1024, 'lr': 0.001, 'optimizer': 'adam', 'epochs': 10},
        {'hidden_dim': 32, 'batch_size': 512, 'lr': 0.005, 'optimizer': 'adam', 'epochs': 10},
        {'hidden_dim': 32, 'batch_size': 512, 'lr': 0.001, 'optimizer': 'adam', 'epochs': 5},
        {'hidden_dim': 32, 'batch_size': 512, 'lr': 0.001, 'optimizer': 'sgd', 'epochs': 10},
        {'hidden_dim': 64, 'batch_size': 512, 'lr': 0.001, 'optimizer': 'sgd', 'epochs': 10},
        {'hidden_dim': 32, 'batch_size': 1024, 'lr': 0.001, 'optimizer': 'sgd', 'epochs': 10},
        {'hidden_dim': 32, 'batch_size': 512, 'lr': 0.005, 'optimizer': 'sgd', 'epochs': 10},
        {'hidden_dim': 32, 'batch_size': 512, 'lr': 0.001, 'optimizer': 'sgd', 'epochs': 5}
    ]
    
    results = []
    
    for i, params in enumerate(tqdm(parameter_combinations, desc="Parameter combinations", unit="config"), 1):
        tqdm.write(f"Running configuration {i}/10: hidden_dim={params['hidden_dim']}, batch_size={params['batch_size']}, lr={params['lr']}, optimizer={params['optimizer']}, epochs={params['epochs']}")
        result = run_multiple_simulations(
            n_runs=10,
            n_individuals=n_individuals,
            n_time_periods=n_time_periods,
            x_dim=x_dim,
            hidden_dim=params['hidden_dim'],
            batch_size=params['batch_size'],
            learning_rate=params['lr'],
            optimizer_name=params['optimizer'],
            n_epochs=params['epochs']
        )
        results.append({
            'Hidden Dim': params['hidden_dim'],
            'Batch Size': params['batch_size'],
            'Learning Rate': params['lr'],
            'Optimizer': params['optimizer'],
            'Epochs': params['epochs'],
            'KS Statistic': result['ks_statistic'],
            'JS Divergence': result['js_divergence'],
            'NRMSE': result['nrmse']
        })
    
    df = save_results_to_csv(results)
    latex_table, latex_table_nrmse = generate_latex_tables(df)
    save_latex_tables(latex_table, latex_table_nrmse)
    
    print("\nSummary Table (Mean of 10 Simulations):")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
