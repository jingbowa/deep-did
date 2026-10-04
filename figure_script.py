import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def evaluate_distributions(true_beta, estimated_beta):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='ks_2samp: Exact calculation unsuccessful. Switching to method=asymp.', category=RuntimeWarning)
        ks_statistic, _ = stats.ks_2samp(true_beta, estimated_beta)
    
    hist_true, bin_edges = np.histogram(true_beta, bins=100, density=True)
    hist_est, _ = np.histogram(estimated_beta, bin_edges, density=True)
    
    hist_true = hist_true + 1e-10
    hist_est = hist_est + 1e-10
    
    hist_true = hist_true / np.sum(hist_true)
    hist_est = hist_est / np.sum(hist_est)
    
    js_divergence = jensenshannon(hist_true, hist_est)
    mse = np.mean((true_beta - estimated_beta)**2)
    variance_true_beta = np.var(true_beta)
    nmse = mse / (variance_true_beta + 1e-10)
    nrmse = np.sqrt(nmse)
    
    return ks_statistic, js_divergence, nrmse

def evaluate_distributions_for_mse(true_beta, estimated_beta):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='ks_2samp: Exact calculation unsuccessful. Switching to method=asymp.', category=RuntimeWarning)
        ks_statistic, _ = stats.ks_2samp(true_beta, estimated_beta)
    
    hist_true, bin_edges = np.histogram(true_beta, bins=100, density=True)
    hist_est, _ = np.histogram(estimated_beta, bin_edges, density=True)
    
    hist_true = hist_true + 1e-10
    hist_est = hist_est + 1e-10
    
    hist_true = hist_true / np.sum(hist_true)
    hist_est = hist_est / np.sum(hist_est)
    
    js_divergence = jensenshannon(hist_true, hist_est)
    mse = np.mean((true_beta - estimated_beta)**2)
    
    return ks_statistic, js_divergence, mse

def plot_data_2():
    results = load_data('./simulated_data/data_2_across_beta_fe.txt')
    
    beta_funcs = results['beta_funcs']
    fixed_effect_funcs = results['fixed_effect_funcs']
    
    n_rows = len(beta_funcs)
    n_cols = len(fixed_effect_funcs)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows), squeeze=False)
    handles, labels = None, None
    
    for i, beta_func in enumerate(beta_funcs):
        for j, fe_func in enumerate(fixed_effect_funcs):
            key = f'beta_{beta_func}_fe_{fe_func}'
            current_result = results['results'][key]
            beta = np.array(current_result['beta'])
            beta_est = np.array(current_result['beta_est'])
            
            ks_statistic, js_divergence, nrmse = evaluate_distributions(beta, beta_est)
            
            ax = axes[i, j]
            ax.hist(beta, bins=100, density=True,
                    histtype="step", linewidth=2.5, color="goldenrod", label="True Beta")
            ax.hist(beta_est, bins=100, density=True,
                    histtype="stepfilled", hatch="//", alpha=0.7,
                    linewidth=0, edgecolor="black", facecolor="steelblue", label="Estimated Beta")
            
            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()

            title_beta_part = beta_func.replace('_', ' ').title()
            title_fe_part = fe_func.replace('_', ' ').title()
            ax.set_title(rf"$\beta(\cdot)$: {title_beta_part}; $f(\cdot, \cdot)$: {title_fe_part}", fontsize=24)
            ax.set_xlabel('Beta Values', fontsize=22)
            ax.set_ylabel('Density', fontsize=22)
            ax.text(0.02, 0.97, f"KS Statistic: {ks_statistic:.3f}\nJS Divergence: {js_divergence:.3f}\nNRMSE: {nrmse:.3f}",
                    transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.tick_params(axis='both', which='major', labelsize=18)
    
    if handles and labels:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=22)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig("./figures/figure_2.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_data_3a():
    results = load_data('./simulated_data/data_3a_across_N_T.txt')
    
    individual_counts = results['individual_counts']
    time_periods = results['time_periods']
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), squeeze=False)
    handles, labels = None, None
    
    for j, t in enumerate(time_periods):
        for i, n in enumerate(individual_counts):
            beta = np.array(results['results'][f'n_{n}_t_{t}']['beta'])
            beta_est = np.array(results['results'][f'n_{n}_t_{t}']['beta_est'])
            
            ks_statistic, js_divergence, nrmse = evaluate_distributions(beta, beta_est)
            
            ax = axes[j, i]
            ax.hist(beta, bins=100, density=True,
                    histtype="step", linewidth=2.5, color="goldenrod", label="True Beta")
            ax.hist(beta_est, bins=100, density=True,
                    histtype="stepfilled", hatch="//", alpha=0.7,
                    linewidth=0, edgecolor="black", facecolor="steelblue", label="Estimated Beta")
            
            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
            
            ax.set_title(f"N: {n:,}, T: {t}", fontsize=24)
            ax.set_xlabel('Beta Values', fontsize=22)
            ax.set_ylabel('Density', fontsize=22)
            ax.text(0.02, 0.97, f"KS Statistic: {ks_statistic:.3f}\nJS Divergence: {js_divergence:.3f}\nNRMSE: {nrmse:.3f}",
                    transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.tick_params(axis='both', which='major', labelsize=18)
    
    if handles and labels:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=22)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig("./figures/figure_3a.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_data_3b():
    results = load_data('./simulated_data/data_3b_across_N_D.txt')

    individual_counts = results['individual_counts']
    x_dims = results['x_dims']
    n_rows = len(x_dims)
    n_cols = len(individual_counts)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows), squeeze=False)
    handles, labels = None, None
    
    for j, x_dim in enumerate(x_dims):
        for i, n in enumerate(individual_counts):
            beta = np.array(results['results'][f'n_{n}_x_{x_dim}']['beta'])
            beta_est = np.array(results['results'][f'n_{n}_x_{x_dim}']['beta_est'])
            
            ks_statistic, js_divergence, nrmse = evaluate_distributions(beta, beta_est)
            
            ax = axes[j, i]
            ax.hist(beta, bins=100, density=True,
                    histtype="step", linewidth=2.5, color="goldenrod", label="True Beta")
            ax.hist(beta_est, bins=100, density=True,
                    histtype="stepfilled", hatch="//", alpha=0.7,
                    linewidth=0, edgecolor="black", facecolor="steelblue", label="Estimated Beta")
            
            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
            
            ax.set_title(f"N: {n:,}, J: {x_dim}", fontsize=24)
            ax.set_xlabel('Beta Values', fontsize=22)
            ax.set_ylabel('Density', fontsize=22)
            ax.text(0.02, 0.97, f"KS Statistic: {ks_statistic:.3f}\nJS Divergence: {js_divergence:.3f}\nNRMSE: {nrmse:.3f}",
                    transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.tick_params(axis='both', which='major', labelsize=18)
    
    if handles and labels:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=22)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig("./figures/figure_3b.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_data_4():
    results = load_data('./simulated_data/data_4_shift_in_fixed.txt')
    
    scenarios = {
        1: "Individual_FE_Shifted",
        3: "Both_FEs_Shifted"
    }
    
    for scenario_num in [1, 3]:
        fig, ax = plt.subplots(figsize=(11.5, 6))
        
        true_beta = np.array(results['results'][f'scenario_{scenario_num}']['true_beta'])
        estimated_beta = np.array(results['results'][f'scenario_{scenario_num}']['estimated_beta'])
        
        ks_statistic, js_divergence, nrmse = evaluate_distributions(true_beta, estimated_beta)
        
        ax.hist(true_beta, bins=100, density=True,
                histtype="step", linewidth=2.5, color="goldenrod", label="True Beta")
        ax.hist(estimated_beta, bins=100, density=True,
                histtype="stepfilled", hatch="//", alpha=0.7,
                linewidth=0, edgecolor="black", facecolor="steelblue", label="Estimated Beta")
        
        ax.legend(fontsize=22, loc='upper right')
        ax.set_xlabel('Beta Values', fontsize=22)
        ax.set_ylabel('Density', fontsize=22)
        ax.text(0.02, 0.97, f"KS Statistic: {ks_statistic:.3f}\nJS Divergence: {js_divergence:.3f}\nNRMSE: {nrmse:.3f}",
                transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        plt.tight_layout()
        if scenario_num == 1:
            filename_suffix = 'a'
        elif scenario_num == 3:
            filename_suffix = 'b'
        else:
            filename_suffix = str(scenario_num)
            
        plt.savefig(f"./figures/figure_4{filename_suffix}.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_data_5a():
    results = load_data('./simulated_data/data_5a_beta_misspecified.txt')
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    handles, labels = None, None
    
    true_beta_types = ['Linear', 'LinearAll', 'Nonlinear']
    beta_layers = ['Linear', 'Nonlinear']
    
    for i, beta_layer in enumerate(beta_layers):
        for j, true_beta_type in enumerate(true_beta_types):
            key = f"{true_beta_type}_{beta_layer}"
            true_beta = np.array(results[key]['true_beta'])
            estimated_beta = np.array(results[key]['estimated_beta'])
            ks_statistic, js_divergence, nrmse = evaluate_distributions(true_beta, estimated_beta)
            
            ax = axes[i, j]
            ax.hist(true_beta, bins=100, density=True,
                    histtype="step", linewidth=2.5, color="goldenrod", label="True Beta")
            ax.hist(estimated_beta, bins=100, density=True,
                    histtype="stepfilled", hatch="//", alpha=0.7,
                    linewidth=0, edgecolor="black", facecolor="steelblue", label="Estimated Beta")
            
            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
            
            model_label = "Linear" if beta_layer == 'Linear' else "DNN"
            ax.set_title(r"True $\beta(\cdot)$: " + f"{true_beta_type},  " + r"$\beta(\cdot)$ Model: " + f"{model_label}", fontsize=24)
            ax.set_xlabel('Beta Values', fontsize=22)
            ax.set_ylabel('Density', fontsize=22)
            ax.text(0.02, 0.97, f"KS Statistic: {ks_statistic:.3f}\nJS Divergence: {js_divergence:.3f}\nNRMSE: {nrmse:.3f}",
                    transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.tick_params(axis='both', which='major', labelsize=18)
    
    if handles and labels:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=22)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig("./figures/figure_5a.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_data_5b():
    results = load_data('./simulated_data/data_5b_fe_misspecified.txt')
    
    interactions = ['additive', 'multiplicative']
    models = ['LinearFE', 'NN']
    n_epochs_val = 10
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    handles, labels = None, None
    
    for i, model in enumerate(models):
        for j, interaction in enumerate(interactions):
            ax = axes[i, j]
            
            key = f"{interaction}_{model}_{n_epochs_val}"
            
            true_beta = np.array(results[key]['true_beta'])
            estimated_beta = np.array(results[key]['estimated_beta'])
            ks_statistic, js_divergence, nrmse = evaluate_distributions(true_beta, estimated_beta)
            
            ax.hist(true_beta, bins=100, density=True,
                    histtype="step", linewidth=2.5, color="goldenrod", label="True Beta")
            ax.hist(estimated_beta, bins=100, density=True,
                    histtype="stepfilled", hatch="//", alpha=0.7,
                    linewidth=0, edgecolor="black", facecolor="steelblue", label="Estimated Beta")
            
            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
            
            if interaction == 'additive':
                interaction_label = "Additive"
            elif interaction == 'multiplicative':
                interaction_label = "Multiplicative"
            else:
                interaction_label = interaction.capitalize()
            
            model_label = "Linear" if model == 'LinearFE' else "DNN"
            
            ax.set_title(r"True $f(\cdot, \cdot)$: " + f"{interaction_label},  " + r"$f(\cdot, \cdot)$ Model: " + f"{model_label}", fontsize=24)
            ax.set_xlabel('Beta Values', fontsize=22)
            ax.set_ylabel('Density', fontsize=22)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.text(0.02, 0.97, 
                    f"KS Statistic: {ks_statistic:.3f}\nJS Divergence: {js_divergence:.3f}\nNRMSE: {nrmse:.3f}",
                    transform=ax.transAxes, fontsize=22, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    if handles and labels:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=22)
    
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig("./figures/figure_5b.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_data_6():
    results = load_data('./simulated_data/data_6_comparison_with_rf.txt')
    
    y_true = np.array(results['y_test'])
    y_pred_nn = np.array(results['y_pred_nn'])
    y_pred_rf = np.array(results['y_pred_rf'])

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    methods_data = [
        ("True Distribution", y_true, y_true), 
        ("DNN Estimator", y_pred_nn, y_true), 
        ("Random Forests Estimator", y_pred_rf, y_true)
    ]

    for i, (method, data_to_plot, true_data_for_comparison) in enumerate(methods_data):
        ax = axes[i]
        
        ax.hist(true_data_for_comparison, bins=100, density=True, 
               histtype="step", linewidth=2.5, color="goldenrod", label='True')
        if method != "True Distribution":
            ax.hist(data_to_plot, bins=100, density=True, 
                   histtype="stepfilled", hatch="//", alpha=0.7,
                   linewidth=0, edgecolor="black", facecolor="steelblue", label='Predicted')
        
        ax.set_title(f"{method}", fontsize=24)
        ax.set_xlabel("Values", fontsize=22)
        ax.set_ylabel("Density", fontsize=22)
        ax.legend(fontsize=22, loc='upper right')
        
        if method != "True Distribution":
            ks_statistic, js_divergence, nrmse = evaluate_distributions(true_data_for_comparison, data_to_plot)
            stats_text = f"KS Statistic: {float(ks_statistic):.3f}\nJS Divergence: {float(js_divergence):.3f}\nNRMSE: {float(nrmse):.3f}"
            ax.text(0.02, 0.95, stats_text,
                    transform=ax.transAxes, fontsize=22, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        ax.tick_params(axis='both', which='major', labelsize=18)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("./figures/figure_6.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    plot_data_2()
    plot_data_3a()
    plot_data_3b()
    plot_data_4()
    plot_data_5a()
    plot_data_5b()
    plot_data_6() 