import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
import os

# Import your modules
import utils
import sofm

def run_synthetic_study():
    sample_sizes = [200, 1000, 5000, 10000]
    n_replications = 20
    results = []

    print(f"Starting Synthetic Study: {len(sample_sizes)} sample sizes, {n_replications} reps each.")
    
    for n_samples in sample_sizes:
        print(f"\n{'='*40}")
        print(f"Running Sample Size: N = {n_samples}")
        print(f"{'='*40}")
        
        for rep in range(n_replications):
            start_time = time.time()
            
            # 1. Generate Data
            da, lam_lat, lam_lon, U_true, L_true, sigma_true = utils._gen_spat_da(
                sq_len=10, 
                k=3, 
                n_samples=n_samples, 
                max_lag=20, 
                ls1=1, 
                ls2=1, 
                mode='gradient',
                sigma = 5
            )
            
            # 2. Initialize and Fit Model
            # Note: Ensure block_sz matches your init parameter name (block_sz vs block_sz)
            mod = sofm.SOFM(da, n_components=3, nonstationary=True, block_sz=1, n_blocks=40)
            mod.fit()
            
            # 3. Calculate Loss for U (Accounting for Sign Ambiguity)
            U_est = mod.U_.detach().numpy() if torch.is_tensor(mod.U_) else mod.U_
            U_true_np = U_true.detach().numpy() if torch.is_tensor(U_true) else U_true
            
            U_loss = 0.0
            for k in range(U_est.shape[1]):
                # Check MSE for both standard and flipped signs
                mse_plus = np.mean((U_est[:, k] - U_true_np[:, k])**2)
                mse_minus = np.mean((U_est[:, k] + U_true_np[:, k])**2)
                U_loss += min(mse_plus, mse_minus)
            U_loss /= U_est.shape[1] # Average MSE across the k components
            
            # 4. Calculate Loss for L
            L_est = mod.L_.detach().numpy() if torch.is_tensor(mod.L_) else mod.L_
            L_true_np = L_true.detach().numpy() if torch.is_tensor(L_true) else L_true
            L_loss = np.mean((L_est - L_true_np)**2)
            
            # 5. Calculate Loss for Sigma
            # Assuming mod.sigma2_ is variance, we take sqrt to compare to sigma_true
            sigma_est = np.sqrt(mod.sigma2_) if not torch.is_tensor(mod.sigma2_) else torch.sqrt(mod.sigma2_).item()
            sigma_true_val = sigma_true.item() if torch.is_tensor(sigma_true) else sigma_true
            sigma_loss = (sigma_est - sigma_true_val)**2
            
            run_time = time.time() - start_time
            print(f"  Rep {rep+1}/{n_replications} | Time: {run_time:.1f}s | U_MSE: {U_loss:.5f} | L_MSE: {L_loss:.2f}")
            
            # Store results
            results.append({
                'Sample_Size': n_samples,
                'Replication': rep + 1,
                'U_MSE': U_loss,
                'L_MSE': L_loss,
                'Sigma_MSE': float(sigma_loss)
            })

    # Save to CSV
    df_results = pd.DataFrame(results)
    csv_filename = "sofm_synthetic_results.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\nStudy complete! Results saved to {csv_filename}")
    
    return df_results

def plot_consistency(df_results):
    """Generates boxplots to visualize consistency (variance shrinking as N increases)."""
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot U Loss
    sns.boxplot(data=df_results, x='Sample_Size', y='U_MSE', ax=axes[0], palette="Blues", showfliers=False)
    sns.stripplot(data=df_results, x='Sample_Size', y='U_MSE', ax=axes[0], color='black', alpha=0.5, size=4)
    axes[0].set_title('Estimation Error: Loadings (U)')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_xlabel('Sample Size (T)')

    # Plot L Loss
    sns.boxplot(data=df_results, x='Sample_Size', y='L_MSE', ax=axes[1], palette="Greens", showfliers=False)
    sns.stripplot(data=df_results, x='Sample_Size', y='L_MSE', ax=axes[1], color='black', alpha=0.5, size=4)
    axes[1].set_title('Estimation Error: Eigenvalues (L)')
    axes[1].set_ylabel('Mean Squared Error')
    axes[1].set_xlabel('Sample Size (T)')

    # Plot Sigma Loss
    sns.boxplot(data=df_results, x='Sample_Size', y='Sigma_MSE', ax=axes[2], palette="Oranges", showfliers=False)
    sns.stripplot(data=df_results, x='Sample_Size', y='Sigma_MSE', ax=axes[2], color='black', alpha=0.5, size=4)
    axes[2].set_title('Estimation Error: Noise Std (Sigma)')
    axes[2].set_ylabel('Squared Error')
    axes[2].set_xlabel('Sample Size (T)')

    plt.tight_layout()
    plt.savefig("sofm_consistency_plots.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df = run_synthetic_study()
    plot_consistency(df)