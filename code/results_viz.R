library(tidyverse)

rm(list = ls())
gc()

# Getting all results in the same shape ----------------------------------------------

files_to_c = list.files("../../results/svm", pattern = "ypred", recursive = T, full.names = T)
for(i in 1:length(files_to_c)){
  res = read.table(files_to_c[[i]],header = T, sep = " ", skip = 0)
  if(unique(res$cenario) == "Cenario_9"){
    res = res %>% mutate(concat_coord = concat_cen9)
    write.table(res, file = files_to_c[[i]] )
    
  }
}


files_tbmodel = list.files('../../data/',
                           full.names = T,
                           pattern = ".RData")

dat = get(load(files_tbmodel[[4]]))
test_lines = which(dat$data %in% unique(dat$data)[8:10])
ver = dat[test_lines,] %>% arrange(data,hora)
concat_cen9 = ver$concat_coord



# Compiling all results to see --------------------------------------------

files_to_viz = list.files("../../results/", pattern = "ypred", recursive = T, full.names = T)
res =list()
for(i in 1:length(files_to_viz)){
  res[[i]] = read.table(files_to_viz[i],header = T, sep = " ", skip = 0) %>% 
    mutate_if(is.factor, as.character)
}  
results = do.call(rbind, res) %>% mutate(erro = ypred - yobs) %>% 
  mutate(abs_error = abs(erro))


# Plotting MAE progress over Scenarios -----------------------------------
res1 = results %>% group_by(tecnica,target,cenario, hmlook_back) %>% 
  summarise(mae = mean(abs_error)) %>% as.data.frame() %>% 
  rename(Técnica = tecnica) %>% 
  mutate(Técnica = case_when(
    Técnica == "ann_lstm" ~ "LSTM",
    Técnica == "svm" ~ "SVR",
    Técnica == "brt" ~ "BRT"
  )) %>% 
  mutate(cenario = case_when(
    cenario == "Cenario_1" ~ "Cenário 1",
    cenario == "Cenario_5" ~ "Cenário 5",
    cenario == "Cenario_7" ~ "Cenário 7",
    cenario == "Cenario_9" ~ "Cenário 9"
  ))

if(target == "temp"){
  #br = 0.5
  y_ax = 'MAE (°C)'
}else{
  #br = 2
  y_ax = 'MAE (%)'
}

(p1 = res1 %>% filter(target == "ur") %>% 
    ggplot(aes(x = hmlook_back, y = mae, col = Técnica, shape = Técnica)) +
    geom_point(size = 4)+
    facet_wrap(.~cenario)+
    theme_bw() + theme(text = element_text(size = 12)) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
    theme(axis.text.x = element_text(size = 14))+
    theme(axis.title.x = element_text(margin = margin(t = 8)))+
    theme(axis.text.y = element_text(size=14))+
    theme(axis.title.y = element_text(margin = margin(r = 8)))+
    theme(axis.title = element_text(size=20))+
    theme(legend.text=element_text(size=16))+
    theme(legend.title = element_text(size = 18))+
    theme(strip.text = element_text(size = 18)) + 
    scale_x_discrete(name ="Intervalo de tempo (min)",
                     limits=c("10","20","30","40","50","60")) +
    ylab(y_ax))

ggsave(filename = paste0("../../part_1/figures/2ndpart_",target,"_",
                         "error_time",".png"), plot = p1,
       dpi = 1200, width = 16, height = 14)

# Plotting ypred x yobs in test set ---------------------------------------

#Fits ok for the variability_part1
res2 = results %>% 
  filter(target == "temp",
         hmlook_back == 1, tecnica == "brt") %>% 
  arrange(data,hora) %>% group_by(concat_coord) %>% 
  mutate(timestamp = seq(1:length(yobs))) %>%
  gather(tipo_medida, medida, c(yobs,ypred)) %>%
  as.data.frame()

res2 %>% ggplot(aes(x = timestamp, y = medida, col = tipo_medida)) + geom_line() + 
  facet_wrap(~cenario, ncol = 2, scales = "free") + xlab('Timestamp') + 
  ylab('Temperatura (°C)') +
  scale_color_manual(labels = c("Real", "Predito"),values=c(rgb(1,0,0), rgb(0,0,1,0.7))) +
  theme_bw() + theme(text = element_text(size = 7)) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
  theme(axis.text.x = element_text(size = 6))+
  theme(axis.text.y = element_text(size=6))+
  theme(axis.title = element_text(size=12))+
  theme(legend.text=element_text(size=10))+
  theme(legend.title = element_text(size = 10))+
  theme(strip.text = element_text(size = 10))
