# ==============================================================================
# COVID-19 Cross Border Modeling - Cluster Analysis
# R Script for PCA and K-means Clustering Analysis
# V.1.0 [2025-08-01]
# Author :: Vidhyakorn Mahd-Adam
# Contact :: mvidhyagorn@gmail.com
# ==============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(factoextra)
  library(cluster)
  library(viridis)
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
})

# Create output directory if it doesn't exist
if (!dir.exists("plots")) {
  dir.create("plots")
}

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================

# Load and prepare dataset
cat("Loading and preparing data...\n")
df <- read.csv("covid_dataset.csv", header = TRUE)
cat("Dataset dimensions:", dim(df), "\n")
head(df)

# Standardize the dataset (columns 2, 3, 4)
df_cleaned <- scale(df[, c(2, 3, 4)])
rownames(df_cleaned) <- df[, 1]  # Preserve country names as row names

# ==============================================================================
# 2. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ==============================================================================

cat("Performing PCA...\n")
# Perform PCA on the standardized dataset
pca_result <- prcomp(df_cleaned, center = FALSE, scale. = FALSE)
summary(pca_result)

# Plot the proportion of variance explained by each PC
p1 <- fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 100), 
               fill = viridis(1)[1], color = viridis(1)[1],
               main = "Scree Plot: Variance Explained by PCs") +
  theme_bw()

# Plot PCA biplot showing individuals and variables
p2 <- fviz_pca_biplot(pca_result, 
                      col.ind = "cos2",
                      gradient.cols = viridis(3, option = "plasma"),
                      col.var = "contrib",
                      repel = TRUE,
                      title = "PCA Biplot: Variables and Individuals") +
  theme_bw()

# Plot individuals on PCA dimensions
p3 <- fviz_pca_ind(pca_result,
                   col.ind = "cos2",
                   gradient.cols = viridis(3, option = "viridis"),
                   repel = TRUE,
                   title = "PCA: Individual Countries") +
  theme_bw()

# Combine contribution plots for PC1 and PC2
contrib_pc1 <- fviz_contrib(pca_result, choice = "var", axes = 1, top = 10)$data
contrib_pc2 <- fviz_contrib(pca_result, choice = "var", axes = 2, top = 10)$data

contrib_pc1$PC <- "PC1"
contrib_pc2$PC <- "PC2"
combined_contrib <- rbind(contrib_pc1, contrib_pc2)

p4 <- ggplot(combined_contrib, aes(x = reorder(name, contrib), y = contrib)) +
  geom_col(fill = viridis(1)[1], alpha = 0.8) +
  geom_hline(yintercept = 100/nrow(contrib_pc1), linetype = "dashed", 
             color = "red", alpha = 0.7) +
  coord_flip() +
  facet_wrap(~ PC, scales = "free_y", nrow = 1) +
  labs(x = "Variables", y = "Contribution (%)", 
       title = "Variable Contributions to Principal Components") +
  theme_bw() +
  theme(
    strip.text = element_text(size = 12, face = "bold"),
    strip.background = element_rect(fill = "lightgray"),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  )

# ==============================================================================
# 3. OPTIMAL CLUSTER DETERMINATION
# ==============================================================================

cat("Determining optimal number of clusters...\n")

# Extract PCA coordinates for clustering
pca_coords <- pca_result$x[, 1:2]

# Create combined plot for all three optimal cluster methods
elbow_data <- fviz_nbclust(df_cleaned, kmeans, method = "wss", k.max = 10)$data
silhouette_data <- fviz_nbclust(df_cleaned, kmeans, method = "silhouette", k.max = 10)$data
gap_data <- fviz_nbclust(df_cleaned, kmeans, method = "gap_stat", k.max = 10)$data

# Standardize data frames
elbow_clean <- data.frame(
  clusters = elbow_data$clusters,
  Value = elbow_data$y,
  Method = "Elbow Method (WSS)"
)

silhouette_clean <- data.frame(
  clusters = silhouette_data$clusters,
  Value = silhouette_data$y,
  Method = "Silhouette Method"
)

gap_clean <- data.frame(
  clusters = gap_data$clusters,
  Value = gap_data$y,
  Method = "Gap Statistic Method"
)

combined_data <- rbind(elbow_clean, silhouette_clean, gap_clean)

p5 <- ggplot(combined_data, aes(x = clusters, y = Value)) +
  geom_line(aes(group = Method), color = "steelblue", size = 1) +
  geom_point(color = "steelblue", size = 2) +
  geom_vline(xintercept = 3, linetype = "dashed", color = "red", alpha = 0.7) +
  facet_wrap(~ Method, scales = "free_y", nrow = 1) +
  labs(x = "Number of Clusters (k)", y = "Value",
       title = "Optimal Number of Clusters - Method Comparison") +
  theme_bw() +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    strip.background = element_rect(fill = "lightgray"),
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold")
  )

# ==============================================================================
# 4. K-MEANS CLUSTERING
# ==============================================================================

cat("Performing k-means clustering...\n")

# Perform k-means clustering with k = 3
set.seed(123)  # For reproducibility
km <- kmeans(df_cleaned, centers = 3, nstart = 25)
km_pca <- kmeans(pca_coords, centers = 3, nstart = 25)

# Plot clustering results on original data
p6 <- fviz_cluster(km, data = df_cleaned,
                   geom = "text",
                   ellipse.type = "convex",
                   ggtheme = theme_bw(),
                   main = "K-means Clustering: Original Data Space",
                   labelsize = 8,
                   palette = viridis(3, option = "viridis")) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12))

# Plot clustering results on PCA coordinates
p7 <- fviz_cluster(km_pca, data = pca_coords,
                   geom = "text",
                   ellipse.type = "convex",
                   ggtheme = theme_bw(),
                   main = "K-means Clustering: PCA Space",
                   labelsize = 8,
                   palette = viridis(3, option = "plasma")) + 
  theme(plot.title = element_text(hjust = 0.5, size = 12))

# ==============================================================================
# 5. CLUSTER COMPARISON AND VALIDATION
# ==============================================================================

# Add cluster assignments to original data frame
df$cluster_original <- km$cluster
df$cluster_pca <- km_pca$cluster

# Display cluster comparison
cat("Cluster comparison table:\n")
cluster_table <- table(df$cluster_original, df$cluster_pca)
print(cluster_table)

# Create comparison plot
p8 <- ggplot(df, aes(x = factor(cluster_original), y = factor(cluster_pca))) +
  geom_count(aes(color = after_stat(n)), size = 5) +
  scale_color_viridis_c(name = "Count", option = "viridis") +
  labs(x = "Original Data Clusters", 
       y = "PCA Space Clusters",
       title = "Clustering Results Comparison") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

# ==============================================================================
# 6. SAVE RESULTS
# ==============================================================================

cat("Saving plots and results...\n")

# Save individual plots
ggsave("plots/01_pca_scree_plot.png", p1, width = 10, height = 6, dpi = 300)
ggsave("plots/02_pca_biplot.png", p2, width = 10, height = 8, dpi = 300)
ggsave("plots/03_pca_individuals.png", p3, width = 10, height = 8, dpi = 300)
ggsave("plots/04_variable_contributions.png", p4, width = 12, height = 6, dpi = 300)
ggsave("plots/05_optimal_clusters_comparison.png", p5, width = 15, height = 5, dpi = 300)
ggsave("plots/06_kmeans_original_space.png", p6, width = 10, height = 8, dpi = 300)
ggsave("plots/07_kmeans_pca_space.png", p7, width = 10, height = 8, dpi = 300)
ggsave("plots/08_cluster_comparison.png", p8, width = 8, height = 6, dpi = 300)

# Save summary results
sink("plots/cluster_analysis_summary.txt")
cat("=== CLUSTER ANALYSIS SUMMARY ===\n\n")
cat("PCA Summary:\n")
print(summary(pca_result))
cat("\nK-means Clustering Results (Original Space):\n")
print(km)
cat("\nK-means Clustering Results (PCA Space):\n")
print(km_pca)
cat("\nCluster Assignment Comparison:\n")
print(cluster_table)
sink()

# Display plots
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
print(p6)
print(p7)
print(p8)

cat("Analysis complete! All plots and results saved to 'plots/' directory.\n")



