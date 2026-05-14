library(mclust)
U_weighted <- read.csv('./U_weighted.csv', header = FALSE)
ground_truth_df <- read.csv('~/Documents/SOFM/ground_truth_labels_151675.csv')
ground_truth <- as.character(ground_truth_df$layer)
mclust_model <- Mclust(U_weighted,G=5:7)
label_pred <- mclust_model$classification
valid_idx <- !is.na(ground_truth) & ground_truth != "NA" & ground_truth != "nan"
ari_score <- adjustedRandIndex(ground_truth[valid_idx], label_pred[valid_idx])
cat(sprintf("Adjusted Rand Index: %.3f\n", ari_score))

library(mclust)
U_weighted <- read.csv('./U_unweighted.csv', header = FALSE)
ground_truth_df <- read.csv('~/Documents/SOFM/ground_truth_labels_151675.csv')
ground_truth <- as.character(ground_truth_df$layer)
mclust_model <- Mclust(U_weighted,G=5:7)
label_pred <- mclust_model$classification
valid_idx <- !is.na(ground_truth) & ground_truth != "NA" & ground_truth != "nan"
ari_score <- adjustedRandIndex(ground_truth[valid_idx], label_pred[valid_idx])
cat(sprintf("Adjusted Rand Index: %.3f\n", ari_score))

library(mclust)
U_weighted <- read.csv('./U_weighted_sq.csv', header = FALSE)
ground_truth_df <- read.csv('~/Documents/SOFM/ground_truth_labels_151675.csv')
ground_truth <- as.character(ground_truth_df$layer)
mclust_model <- Mclust(U_weighted,G=5:7)
label_pred <- mclust_model$classification
valid_idx <- !is.na(ground_truth) & ground_truth != "NA" & ground_truth != "nan"
ari_score <- adjustedRandIndex(ground_truth[valid_idx], label_pred[valid_idx])
cat(sprintf("Adjusted Rand Index: %.3f\n", ari_score))

library(mclust)
U_weighted <- read.csv('./U_stacked.csv', header = FALSE)
ground_truth_df <- read.csv('~/Documents/SOFM/ground_truth_labels_151675.csv')
ground_truth <- as.character(ground_truth_df$layer)
mclust_model <- Mclust(U_weighted,G=5:7)
label_pred <- mclust_model$classification
valid_idx <- !is.na(ground_truth) & ground_truth != "NA" & ground_truth != "nan"
ari_score <- adjustedRandIndex(ground_truth[valid_idx], label_pred[valid_idx])
cat(sprintf("Adjusted Rand Index: %.3f\n", ari_score))




















library(mclust)
sample_ids = c(151676,151675,151674,151673,151672,151671,151670,151669,151510,151509,151508,151507)
for(sample_id in sample_ids){
	U_weighted <- read.csv(paste0('/projectnb/modislc/users/danc/SpatialCCA/U_weighted_',sample_id,'_ricker.csv'), header = FALSE)
	ground_truth_df <- read.csv(paste0('/projectnb/modislc/users/danc/spatialCCA/ground_truth_labels_',sample_id,'.csv'))
	ground_truth <- as.character(ground_truth_df$layer)
	mclust_model <- Mclust(U_weighted,G=7)
	label_pred <- mclust_model$classification
	valid_idx <- !is.na(ground_truth) & ground_truth != "NA" & ground_truth != "nan"
	ari_score <- adjustedRandIndex(ground_truth[valid_idx], label_pred[valid_idx])
	cat(sprintf("Adjusted Rand Index: %.3f\n", ari_score))	
}

