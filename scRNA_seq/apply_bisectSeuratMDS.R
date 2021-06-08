## Apply Seurat with MDS

suppressPackageStartupMessages({
  library(Seurat)
})

apply_bisectSeuratMDS <- function(sce, params, minresolution, maxresolution, n_clusters) {   #apply bisection method to find right number of cluster
  (seed <- round(1e6*runif(1)))
  tryCatch({
    dat <- counts(sce)
    st <- system.time({
      M <- -1
      k1 <- n_clusters + 1
      k2 <- 0
      res <- NULL
      iterations <- 0
      
      ## Seurat
      dat <- counts(sce)
      pbmc <- CreateSeuratObject(counts = dat, project = "pbmc3k", min.cells = 0, min.features = 0) #pbmc.data
      #pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
      #pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
      pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)
      # HVG
      pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 10000)
      # Scaling the data
      all.genes <- rownames(pbmc)
      pbmc <- ScaleData(pbmc, features = all.genes)
      pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))
      print("Run clustering on MDS")
      pbmc@reductions[["pca"]]@cell.embeddings <- params$mds_data
      row.names(pbmc@reductions[["pca"]]@cell.embeddings) <- colnames(pbmc@assays[["RNA"]]@data)
      # pbmc@reductions[["pca"]]@feature.loadings <- NULL
      # clustering cells
      pbmc <- FindNeighbors(pbmc, dims = 1:ncol(mds_data))
      
      #print(unique(cluster))
      
      cluster <- NULL
      while(iterations < 40 & abs(maxresolution - minresolution) > 1e-6){
        print(paste0("============================================  Min resolution: ", toString(minresolution), ", max resolution: ", toString(maxresolution)))
        if (minresolution != M){
          print(paste0("============================================ Resolution: ", toString(minresolution)))
          # res <- apply_mMDSSeurat(sce = sce, params = params, resolution = minresolution)
          pbmc <- FindClusters(pbmc, resolution = minresolution)
          cluster <- pbmc@active.ident
          iterations <- iterations + 1
          k1 <- length(unique(cluster))
          if (k1 == n_clusters){break}
        }
        
        if (maxresolution != M){
          print(paste0("============================================ Resolution: ", toString(maxresolution)))
          # res <- apply_mMDSSeurat(sce = sce, params = params, resolution = maxresolution)
          pbmc <- FindClusters(pbmc, resolution = maxresolution)
          cluster <- pbmc@active.ident
          iterations <- iterations + 1
          k2 <- length(unique(cluster))
          if (k2 == n_clusters){break}
        }
        # we need to find minresol and maxresol s.t k1 < n_clusters < k2
        while (k1 > n_clusters & iterations < 6){
          maxresolution <- minresolution
          minresolution <- minresolution/2
          print(paste0("============================================ Resolution: ", toString(minresolution)))
          # res <- apply_mMDSSeurat(sce = sce, params = params, resolution = minresolution)
          pbmc <- FindClusters(pbmc, resolution = minresolution)
          cluster <- pbmc@active.ident
          iterations <- iterations + 1
          k1 <- length(unique(cluster))
          if (k1 == n_clusters){break}
        }
        while (k2 < n_clusters & iterations < 6){
          minresolution <- maxresolution
          maxresolution <- 2*maxresolution
          print(paste0("============================================ Resolution: ", toString(maxresolution)))
          # res <- apply_mMDSSeurat(sce = sce, params = params, resolution = maxresolution)
          pbmc <- FindClusters(pbmc, resolution = maxresolution)
          cluster <- pbmc@active.ident
          iterations <- iterations + 1
          k2 <- length(unique(cluster))
        }
        
        # find bisection value
        M <- (minresolution+maxresolution)/2
        print(paste0("============================================ Resolution: ", toString(M)))
        # res <- apply_mMDSSeurat(sce = sce, params = params, resolution = M)
        pbmc <- FindClusters(pbmc, resolution = M)
        cluster <- pbmc@active.ident
        iterations <- iterations + 1
        k3 <- length(unique(cluster))
        if (k3 == n_clusters){
          break
        } else if (k3 < n_clusters){
          minresolution <- M
        } else {
          maxresolution <- M
        }
      }
      
      # cluster <- res$cluster
    })
    
    st <- c(user.self = st[["user.self"]], sys.self = st[["sys.self"]], 
            user.child = st[["user.child"]], sys.child = st[["sys.child"]],
            elapsed = st[["elapsed"]])
    list(st = st, cluster = cluster, est_k = NA, iterations = iterations)
  }, error = function(e) {
    list(st = c(user.self = NA, sys.self = NA, user.child = NA, sys.child = NA,
                elapsed = NA), 
         cluster = structure(rep(NA, ncol(sce)), names = colnames(sce)),
         est_k = NA)
  })
}

