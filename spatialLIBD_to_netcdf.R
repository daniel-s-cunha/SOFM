library(ncdf4)
library(spatialLIBD)
library(SpatialExperiment)
library(here)

setwd('/projectnb/modislc/users/danc/')
output_dir_nc <- here("dlpfc_netcdf_data")
dir.create(output_dir_nc, showWarnings = FALSE)

spe <- fetch_data(type = "spe")
sample_ids <- unique(spe$sample_id)

for (current_sample in sample_ids) {

  spe_subset <- spe[, spe$sample_id == current_sample]

  expression_matrix <- assay(spe_subset, "logcounts")
  
  spatial_coords <- spatialCoords(spe_subset) 
  
  gene_ids <- rowData(spe_subset)$gene_id
  spot_barcodes <- colnames(spe_subset)
  spot_metadata <- colData(spe_subset)

  dim_gene <- ncdim_def(
    name = "gene",
    units = "index",
    vals = 1:nrow(expression_matrix)
  )

  dim_spot <- ncdim_def(
    name = "spot",
    units = "index",
    vals = 1:ncol(expression_matrix)
  )

  dim_char <- ncdim_def(
    name = "max_char_len",
    units = "",
    vals = 1:max(nchar(c(gene_ids, spot_barcodes))),
    create_dimvar = FALSE
  )

  var_logcounts <- ncvar_def(
    name = "logcounts",
    units = "log-normalized expression",
    dim = list(dim_gene, dim_spot),
    prec = "double"
  )
  var_pxl_col <- ncvar_def(
    name = "pxl_col_in_fullres",
    units = "pixels",
    dim = list(dim_spot),
    prec = "integer"
  )
  var_pxl_row <- ncvar_def(
    name = "pxl_row_in_fullres",
    units = "pixels",
    dim = list(dim_spot),
    prec = "integer"
  )

  var_array_col <- ncvar_def(name = "array_col", units = "index", dim = list(dim_spot), prec = "integer")
  var_array_row <- ncvar_def(name = "array_row", units = "index", dim = list(dim_spot), prec = "integer")
  
  var_gene_id <- ncvar_def(
    name = "gene_id",
    units = "Ensembl ID",
    dim = list(dim_char, dim_gene),
    prec = "char"
  )
  var_spot_barcode <- ncvar_def(
    name = "spot_barcode",
    units = "barcode",
    dim = list(dim_char, dim_spot),
    prec = "char"
  )

  nc_filename <- paste0(current_sample, "_spatial_expression_fullres.nc")
  nc_path <- here(output_dir_nc, nc_filename)

  nc_file <- nc_create(nc_path,
                       vars = list(var_logcounts, var_pxl_col, var_pxl_row,
                                   var_array_col, var_array_row,
                                   var_gene_id, var_spot_barcode),
                       force_v4 = TRUE)

  ncvar_put(nc_file, var_logcounts, expression_matrix)  
  
  ncvar_put(nc_file, var_pxl_col, spatial_coords[, "pxl_col_in_fullres"])
  ncvar_put(nc_file, var_pxl_row, spatial_coords[, "pxl_row_in_fullres"])
  
  ncvar_put(nc_file, var_array_col, spot_metadata$array_col)
  ncvar_put(nc_file, var_array_row, spot_metadata$array_row)  
  ncvar_put(nc_file, var_gene_id, gene_ids)
  ncvar_put(nc_file, var_spot_barcode, spot_barcodes)
  
  ncatt_put(nc_file, "logcounts", "coordinates", "pxl_col_in_fullres pxl_row_in_fullres array_col array_row")
  
  ncatt_put(nc_file, 0, "source", "LIBD Human DLPFC 10x Visium")
  ncatt_put(nc_file, 0, "sample_id", current_sample)

  nc_close(nc_file)

}