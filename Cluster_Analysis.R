# Install Packages
install.packages("factoextra")
install.packages("cluster")

library(factoextra)
library(cluster)

# Download dataset
df <- read.csv(~covid_dataset.csv,header = TRUE)
head(df)

# Standardize the dataset
df_cleaned <- df[,c(2,3,4)]
df_cleaned <- scale(df_cleaned)

# Determine numbers of cluster ( result = 3 clusters)
fviz_nbclust(df_cleaned, kmeans, method ="wss")

# Perform k-means clustering with k = 3 clusters
km <- kmeans(df_cleaned, centers = 3, nstart = 25)

# Plot results of final k-means model
fviz_cluster(km, data = df_cleaned)


