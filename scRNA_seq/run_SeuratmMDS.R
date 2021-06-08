rm(list = ls())
library(Seurat)
library(SingleCellExperiment)
library(Rtsne)
library(MASS) # Nonmetric MDS
library(DuoClustering2018)
library(mclust)
source("apply_bisectSeuratMDS.R")

get_ouput_file_name <- function(data_dir)
{
  n <- nchar(data_dir)
  out_dir <- ""
  iter_count <- 0
  for (i in seq(n-4,2,-1)){
    s <- substr(data_dir,i,i)
    if (iter_count==3){
      break 
    }
    if (s=='/'){
      iter_count <- iter_count+1
      out_dir <- paste0('_', out_dir)
    }
    else {
      out_dir <- paste0(s, out_dir)
    }
  }
  out_dir <- substr(out_dir,2,nchar(out_dir))
  return(out_dir)
}
get_last_file_name <- function(data_dir)
{
  n <- nchar(data_dir)
  out_dir <- ""
  iter_count <- 0
  for (i in seq(n-4,2,-1)){
    s <- substr(data_dir,i,i)
    if (iter_count==1){
      break 
    }
    if (s=='/'){
      iter_count <- iter_count+1
      out_dir <- paste0('_', out_dir)
    }
    else {
      out_dir <- paste0(s, out_dir)
    }
  }
  out_dir <- substr(out_dir,2,nchar(out_dir))
  return(out_dir)
}
## Get last 4 letters in the string "file"
last4 <- function(file)
{
  n <- nchar(file)
  return(substr(file,n-3,n))
}
lastk <- function(file, k)
{
  n <- nchar(file)
  return(substr(file,n-k+1,n))
}

# Input mMDS directory which contain the embedding.
mMDS_dir <- "/data/domagoj/csv_subset_l2/"

# Data directory which contain the gene expression matrix (.rds format).
list_dir <- "/data/scRNAseq_clustering_benchmark/data/mMDS/" 

# Output directory
output_dir <- "/data/scRNAseq_clustering_benchmark/results/sc3_seurat_labels_results/"


## MDS for #PCs = 2, 5, 10, 20, 50
params <- duo_clustering_all_parameter_settings_v2()[[paste0("sce_filteredExpr10_Koh_","SC3")]]
for (mds_PCs in c("2", "5", "10", "20", "50")){ #loop number of dimensions in MDS 
# for (mds_PCs in c("2")){
  #method <- paste0("LouvainMDScos", mds_PCs)
  method <- paste0("LouvainMDS_l2", mds_PCs)
  for (file in list.files(list_dir, recursive = T, pattern='.rds'))
  {
    data_dir <- paste0(list_dir,file)
    print("=================================================================================================")
    print(data_dir)
    dataname <- get_last_file_name(data_dir)
    print(dataname)

    sce <- readRDS(data_dir);
    if(!is.null(sce$cell_type1))
    {
      colData(sce)$phenoid <- sce$cell_type1
    }
    labels_true <- colData(sce)$phenoid

    ## Read the MDS output 
    mds_data <- read.csv(paste0(mMDS_dir, dataname, "PCA100_", mds_PCs, ".csv"), header = FALSE)

    params$mds_data <- as.matrix(mds_data)
    minresolution <- 1.0
    maxresolution <- 5.0
    n_clusters <- length(unique(colData(sce)$phenoid))

    res <- apply_bisectSeuratMDS(sce = sce, params = params, minresolution = 1.0, maxresolution = 5.0, n_clusters = n_clusters)

    output_results <- data.frame(method = res$cluster, labels_true = labels_true, time=rep(0, length(labels_true)),
                                 memory = rep(0, length(labels_true)))
    write.table(output_results, paste0(output_dir,get_ouput_file_name(data_dir), method, '_labels.csv'), sep = ",", row.names = F)
    print(adjustedRandIndex(res$cluster, labels_true))
  }
}

