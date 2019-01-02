library(tidyverse)
library(gtools) #for ordering names

rm(list = ls())
gc()

#Verifying all files to be modified

files_tbmod = list.files('../../data_raw/',
                         full.names = T)


# Reading_data ------------------------------------------------------------

#reading and storing all files from files_tbmod
#cleaning columns and creatig concat coordinates for future filtering
df_list = lapply(files_tbmod, 
                 function(x) read_delim(x, delim = ",",col_names = TRUE) %>% 
                   select(-X1) %>% 
                   select(-(Lateral_plastico:Meio_poroso)) %>%
                   mutate(concat_coord = paste0(as.character(x),as.character(y),as.character(z))) %>% 
                   as.data.frame())

# creating look back for time series prediction
look_back = function(df, colNames, lags = 1:6){ #data until one hour back
  n = nrow(df)
  vet_names = colNames
  for(lag in lags){
    for(col in vet_names){
      new_col = paste0(col,"_prev_",lag)
      row_range_in = 1:(n-lag)
      df[,new_col] = c(rep(NA,lag), unname(df[row_range_in,col]))
    }
  }
  return(df)
}

#Function to apply look back
apply_lback = function(df){
  
  #criando campo concatenado para filtro
  unique_coords = unique(df$concat_coord)
  
  dfi = list()
  
  #defining columns to be lagged
  col_lagged = df %>% 
    select(-c(data,hora,medicao, x, y, z, cenario, concat_coord, range_datas)) %>% 
    colnames()

  # for filtering and lagging for each coordinate (on columns previously selected)
  for(xyz in unique_coords){
    dfi[[xyz]] = df %>% filter(concat_coord == xyz)
    dfi[[xyz]] = look_back(dfi[[xyz]], col_lagged)
  }
  
  df = do.call(bind_rows, dfi)
  sortedNames = mixedsort(colnames(df))
  df = df[c(sortedNames)]
  
  save(df, file = paste0("../../data/df",min(df$data),".RData"))
  return(df)
}

a = lapply(df_list, function(x) apply_lback(x))
