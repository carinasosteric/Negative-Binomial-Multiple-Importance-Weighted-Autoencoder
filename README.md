# Negative-Binomial-Multiple-Importance-Weighted-Autoencoder
A PyTorch implementation of a Negative Binomial Missing-data Importance-Weighted Autoencoder (NB-MIWAE) for imputing missing values in bulk RNA-seq count matrices with few samples. Imputed values come back as integer counts that are ready for differential expression analysis with DESeq2.

## Motivation

Most deep imputation methods assume Gaussian noise. RNA-seq counts are discrete, non-negative and overdispersed, so their variance grows faster than their mean. If imputation flattens that mean–variance relationship, DESeq2's dispersion estimates are distorted and downstream conclusions become unreliable.

Autoencoders for RNA-seq data also usually need large sample sizes. Bulk RNA-seq studies often have only a few dozen samples.

NB-MIWAE tackles both problems:

A Negative Binomial likelihood models counts and overdispersion directly, so the imputed data keeps the statistical structure DESeq2 relies on.
Importance-weighted training (MIWAE) draws many latent samples per observation. This gives a tighter bound on the likelihood and makes better use of limited data than a standard VAE.

## Model

Encoder. Takes library-size-normalised, log1p-transformed counts concatenated with the missingness mask. It outputs a Gaussian posterior q(z | x) over a low-dimensional latent space.

Importance sampling. Draws K = 100 latent samples per observation using the reparameterisation trick.

Decoder. Maps each latent sample to a per-gene NB mean and dispersion, predicted in log space so both stay positive.

Objective. Maximises the importance-weighted ELBO:

log w = log p(x_obs | z) + log p(z) − log q(z | x)
loss  = −mean( logsumexp(log w) − log K )

The NB log-likelihood is computed on observed entries only, under an MCAR assumption. A small penalty discourages near-zero predicted means where the observed count is non-zero.

Imputation. The decoded means from L = 100 latent samples are averaged, weighted by their self-normalised importance weights. Only missing entries are filled; observed values are left unchanged. Outputs are rounded to non-negative integers.

### Training.

Adam optimiser (weight decay 1e-5) with ReduceLROnPlateau
Gradient clipping (max norm 0.5) and clamped NB parameters for numerical stability
Batches with NaN or infinite loss are skipped
Early stopping (patience 50) that restores the best weights
Grid search over latent size, hidden size and learning rate
Evaluation

10% of observed entries are held out and treated as missing during training, then compared against their true values.

### Category	Metrics
Accuracy	MAE, RMSE, NRMSE and Spearman correlation (all values, non-zero only, and log space); log MAE stratified by expression quartile
Zero handling	Percentage of true zeros falsely imputed as non-zero (PFC)
Count structure	Mean–variance log-log slope (observed vs imputed), per-gene mean and variance correlation, dispersion stability (genes losing defined dispersion, median log fold change, extreme shifts)

The count-structure metrics test what matters for downstream analysis: whether the imputed data still behaves like NB-distributed counts.

## Results

The model was tested on liver RNA-seq from 39 wild house mice (Mack et al., 2018), with 5%, 10% and 30% of values removed completely at random. About 23,000 genes remained after filtering.

Rank preservation: high correlation with the true held-out values at every missingness level (Spearman ρ > 0.95).
Mean–variance structure: the imputed mean–variance slope closely matched the observed data. It stayed close even at 30% missingness, which a Gaussian-based imputer would tend to flatten.
Dispersion: few genes showed extreme dispersion shifts after imputation.
Zero inflation: about 40% of true zeros were imputed as non-zero. This is the main weakness and a clear target for future work.

## Data format

The input is a whitespace-delimited text file of raw counts:

Rows: genes, with gene IDs in the first column
Columns: samples
Missing values: NA

Genes are kept if they have more than 10 observed values and more than 5 non-zero values.

## Requirements
Python 3.10+ (developed on 3.13)
PyTorch (uses a CUDA GPU if available)
NumPy, pandas, SciPy, scikit-learn, Matplotlib
bash
pip install torch numpy pandas scipy scikit-learn matplotlib
Usage

Final hyperparameters: latent size 32, hidden size 512, K = 100, learning rate 5e-4, batch size 16.

## Outputs:

imputation_metrics_miwae_log_FINAL.csv: evaluation metrics, one row per dataset
true_vs_imputed_log.png: true vs imputed held-out values (log space, non-zero only)
tuned_DESeq2_Ready_miwae_log_<dataset>.csv: imputed integer count matrix (genes × samples) for DESeq2
Limitations and future work
Zero handling. A zero-inflated NB (ZINB) likelihood, or a learned dropout probability, could reduce false non-zero imputations.
Missingness assumption. The model assumes MCAR. Real RNA-seq dropout is often MNAR, with lowly expressed genes more likely to be missing, so this would need explicit modelling.
Sample size. With n = 39, performance depends on careful hyperparameter tuning.
Gene filtering. Removing lowly expressed genes before imputation reduces memory use but also removes information the model could learn from.

## References
Mattei PA, Frellsen J. MIWAE: Deep generative modelling and imputation of incomplete data sets. ICML. 2019.
Burda Y, Grosse R, Salakhutdinov R. Importance weighted autoencoders. ICLR. 2016.
Eraslan G, et al. Single-cell RNA-seq denoising using a deep count autoencoder. Nature Communications. 2019;10:390.
Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biology. 2014;15:550.
Mack KL, Ballinger MA, Phifer-Rixey M, Nachman MW. Gene regulation underlies environmental adaptation in house mice. Genome Research. 2018;28(11):1636–45.
