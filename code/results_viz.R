library(tidyverse)

rm(list = ls())
gc()

# getting BRT results in same format
files_to_c = list.files("../../results/brt", pattern = "ypred", recursive = T, full.names = T)
for(i in 1:length(files_to_c)){
  res = read.table(files_to_viz[[i]],header = T, sep = " ", skip = 0)
  if(unique(res$cenario) == "Cenario_9"){
    res  = res %>% mutate(tecnica = "brt") %>% rename(yobs = truth, ypred = response)
    res = res %>% select(-c(iter,set,id))
    res = res %>% mutate(range_datas = range_cen9, data = data_cen9, hora = hora_cen9,
                         concat_coord = concat_cen9)
    res = res[,c("tecnica","cenario","range_datas","concat_coord","data",
                 "hora","target","hmlook_back","yobs","ypred")]
    write.table(res, file = files_to_viz[[i]] )
    
  }
}







# Compiling all results to see --------------------------------------------

files_to_viz = list.files("../../results/", pattern = "ypred", recursive = T, full.names = T)
res =list()
for(i in 1:length(files_to_viz)){
  res[[i]] = read.table(files_to_viz[[i]],header = T, sep = " ", skip = 0) %>% 
    mutate_if(is.factor, as.character)
}  
results = do.call(bind_rows, res) %>% mutate(erro = ypred - yobs)


# Plotting MAE progress over Scenarios -----------------------------------
res1 = results %>% group_by(tecnica,target,cenario, hmlook_back) %>% summarise(mae = mean(abs(erro)))
res1 %>% filter(target == "temp") %>% 
  ggplot(aes(x = hmlook_back, y = mae, col = cenario)) + geom_line()+
  facet_wrap(.~tecnica)
