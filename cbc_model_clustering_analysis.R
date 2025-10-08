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
  library(ggrepel)
  library(showtext)  # For custom fonts
})

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================

# Load and prepare dataset
df <- read.csv("covid_dataset.csv", header = TRUE)
df <- df %>%
  select(country_label, infec, pos, vac)

# Standardize the dataset (columns 2, 3, 4)
df_cleaned <- scale(df[, c(2, 3, 4)])
rownames(df_cleaned) <- df[, 1]  # Preserve country names as row names

# ==============================================================================
# 2. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ==============================================================================

# Perform PCA on the standardized dataset
pca_result <- prcomp(df_cleaned, center = FALSE, scale. = FALSE)
summary(pca_result)

# Plot the proportion of variance explained by each PC
p1 <- fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 100), 
               fill = viridis(1)[1], color = viridis(1)[1],
               main = "Scree Plot: Variance Explained by PCs") +
  theme_bw(base_family = "inter")

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
  theme_bw(base_family = "inter") +
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
  theme_bw(base_family = "inter") +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    strip.background = element_rect(fill = "lightgray"),
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold")
  )

# ==============================================================================
# 4. K-MEANS CLUSTERING
# ==============================================================================
# Perform k-means clustering with k = 3
set.seed(123)  # For reproducibility
library(ggplot2)
library(colorblindcheck)
library(cols4all)

km <- kmeans(df_cleaned, centers = 3, nstart = 25)
km_pca <- kmeans(pca_coords, centers = 3, nstart = 25)

# Darker viridis palette
pal_dark <- hcl.colors(n = 3, palette = "viridis")

# Centroids in PCA space
pca_df <- data.frame(PC1 = pca_result$x[, 1],
                     PC2 = pca_result$x[, 2],
                     cluster = factor(km$cluster),
                     label = rownames(df_cleaned))

# Plot clustering results on original data (shown in PCA space by fviz_cluster)
p6 <- fviz_cluster(km, data = df_cleaned,
                   geom = "point",
                   ellipse.type = "convex",
                   ggtheme = theme_bw(base_family = "inter"),
                   main = "K-means Clustering: Original Data Space",
                   palette = pal_dark,
                   show.clust.cent = FALSE,
                   pointsize = 2) +
  labs(x = "PC1", y = "PC2", color = NULL, fill = NULL, shape = NULL) +
  scale_color_manual(values = pal_dark,
                     labels = c("Cluster 1", "Cluster 2", "Cluster 3")) +
  scale_fill_manual(values = pal_dark,
                    labels = c("Cluster 1", "Cluster 2", "Cluster 3")) +
  scale_shape_manual(values = c(16, 17, 15),
                     labels = c("Cluster 1", "Cluster 2", "Cluster 3")) +
  guides(color = guide_legend(override.aes = list(size = 3)),
         fill = guide_legend(override.aes = list(size = 3)),
         shape = guide_legend(override.aes = list(size = 3))) +
  theme(legend.position = c(0.05, 0.95),
        legend.justification = c(0, 1),
        legend.background = element_rect(fill = "white", color = "black"),
        legend.margin = margin(5, 5, 5, 5),
        legend.title = element_blank(),                                       # Remove legend title
        plot.title = element_text(hjust = 0.5, size = 12, family = "inter"),
        text = element_text(family = "inter")) +
  ggrepel::geom_text_repel(
    data = pca_df,
    aes(x = PC1, y = PC2, label = label),
    color = "black",
    size = 3,
    family = "inter",
    box.padding = 0.3,
    point.padding = 0.2,
    segment.color = "black",
    segment.size = 0.3,
    max.overlaps = Inf,
    min.segment.length = 0
  )






