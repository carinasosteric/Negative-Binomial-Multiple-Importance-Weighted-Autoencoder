# Negative-Binomial-Multiple-Importance-Weighted-Autoencoder
A PyTorch implementation of a Missing-data Importance-Weighted Autoencoder (MIWAE) with a Negative Binomial (NB) likelihood, built to impute missing values in bulk RNA-seq count matrices. The imputed matrix is returned as integer counts, ready for downstream differential expression analysis with DESeq2.

Most deep imputation methods assume Gaussian noise. That suits continuous data, but RNA-seq counts are discrete, non-negative and overdispersed. Gaussian-based imputation can flatten the mean–variance relationship that tools like DESeq2 depend on. This project models counts with an NB distribution and evaluates imputation on two fronts: how close the imputed values are, and whether the statistical structure DESeq2 relies on survives.


## How it works

Encoder. Takes library-size-normalised, log1p-transformed counts concatenated with the missingness mask, and outputs the mean and log-variance of a Gaussian posterior q(z | x) over a low-dimensional latent space.

Importance sampling. Draws K latent samples per observation using the reparameterisation trick.

Decoder. Maps each latent sample to per-gene NB parameters (mean μ and dispersion r), predicted in log space so both stay positive.

Objective. Maximises the importance-weighted ELBO, computing the NB log-likelihood on observed entries only, so missing values never influence training (MCAR assumption). A small penalty discourages near-zero predicted means where the observed count is non-zero.

Imputation. Draws L latent samples, weights the decoded means by their importance weights (self-normalised), and fills only the missing entries. Observed values are left unchanged. Outputs are rounded and clipped to non-negative integers.

Training details
Adam optimiser (weight decay 1e-5) with ReduceLROnPlateau (factor 0.5, patience 50)
Gradient clipping (max norm 0.5) and clamping of NB parameters for numerical stability
Batches with NaN or infinite loss are skipped
Early stopping (patience 50) restores the best-performing weights
Hyperparameter grid search over latent size, hidden size and learning rate
Evaluation

10% of observed entries are held out (seed 42) and treated as missing during training, then compared against their true values. Metrics are grouped into four categories:

Category	Metrics
Accuracy	MAE, RMSE, NRMSE, Spearman correlation (all values and non-zero only)
Log space	Log MAE/RMSE, Spearman, % of imputations within 0.5 / 1.0 / 2.0 log units
Zero handling	Percentage of true zeros falsely imputed as non-zero (PFC)
Count structure	Mean–variance log-log slope (observed vs imputed), per-gene mean/variance correlation, dispersion stability (genes losing defined dispersion, median log fold change, extreme shifts)
Stratified	Log MAE and Spearman by expression quartile

The count-structure metrics check whether imputation preserves the overdispersion that DESeq2 uses to estimate dispersion, and therefore whether downstream biological conclusions remain trustworthy.

## Requirements
Python 3.10+ (developed on 3.13)
PyTorch (a CUDA GPU is used automatically if available)
NumPy, pandas, SciPy, scikit-learn, Matplotlib
bash
pip install torch numpy pandas scipy scikit-learn matplotlib
Input data format

A whitespace-delimited text file of raw counts:

Rows: genes (first column contains gene IDs)
Columns: samples
Missing values: NA

Genes are kept only if they have more than 10 observed values and more than 5 non-zero values.

## Usage

Each dataset is run separately due to memory constraints.

Set the input file name in the data-loading cell (e.g. countdata5.txt).
Run the cells in order: definitions → data loading → (optional) grid search → training → imputation → evaluation.
Update the dataset_label and output file names where marked # change per dataset.

Outputs:

imputation_metrics_miwae_log_FINAL.csv: one row of evaluation metrics per dataset (appended on each run)
true_vs_imputed_log.png: scatter plot of true vs imputed held-out values (log space, non-zero)
tuned_DESeq2_Ready_miwae_log_<dataset>.csv: imputed integer count matrix (genes × samples) for DESeq2

Final hyperparameters used: latent size 32, hidden size 512, K = 100, learning rate 5e-4, batch size 16.
