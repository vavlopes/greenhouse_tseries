library(tidyverse)
library(arules)
library(lubridate)
library(gtools)

rm(list = ls())
gc()


# Functions ----------------------------------------------------------------
# Function to gather information for 4 teste scenarios

gather_info = function(cenario, target){
  #csv cenario files
  data = list.files("../../part_1/data_raw_part1/", pattern = cenario, 
                    recursive = T, full.names = T)
  
  df1 = read.table(data, row.names = 1, header = T, sep = ",") %>% 
    mutate(z = ifelse(z != 0, z + 1.2, z))
  
  #dados preditos
  predicted = list.files("../../part_1/results_part1/", pattern = paste0(cenario,"_",target),
                         recursive = T, full.names = T)
  pred = list()
  for(i in 1:length(predicted)){
    pred[[i]] = read.table(predicted[i], row.names = 1, header = T)
    pred[[i]] = cbind(pred[[i]],df1) %>% select(predicted,real,tecnica,data,hora,x,y,z)
  }
  
  pred = do.call(bind_rows,pred)
  
  return(pred)
}

# Function to plot REC curves

Rec_plot = function(cenario,target){
  
  pred = gather_info(cenario = cenario, target = target)
  
  # REC Curve ---------------------------------------------------------------
  
  dat = data.frame(ypred = pred$predicted,
                   yobs = pred$real,
                   ae = abs(pred$predicted-pred$real),
                   type = pred$tecnica)
  
  dat$clss = discretize(dat$ae, method = "frequency",breaks = 100)
  breaks = discretize(dat$ae, method = "frequency",breaks = 100, onlycuts = T)
  
  dat = dat %>% group_by(type,clss) %>% count() %>% as.data.frame() %>% 
    group_by(type) %>% 
    mutate(soma = cumsum(n)) %>% 
    mutate(acc = soma/max(soma)) %>% 
    rename(Modelos = type) %>% 
    as.data.frame()
  
  dat$breaks = rep(breaks[2:length(breaks)],3)
  
  #Creting 3 lines to start the graph in zero
  to_bind = data.frame(Modelos = c("brt","svm","rf"),
                       clss = "[0,0)",
                       n = 0,
                       soma = 0,
                       acc = 0,
                       breaks = 0)
  
  dat = rbind(dat,to_bind)
  
  #geting references near 80% of accuracy
  ref = dat %>% mutate(dif = abs(acc - 0.8)) %>% arrange(dif) %>% head(3)
  refy = ref[1,"acc"]
  ref = ref %>% arrange(breaks)
  refx = ref[1,"breaks"]
  x_lim = dat %>% filter(acc >= 0.8, acc <= 0.95) %>% arrange(desc(breaks))
  x_lim = round(x_lim[1,"breaks"],0) + 1
  
  if(target == "temp"){
    br = 0.5
    x_ax = "Erro Absoluto (°C)"
  }else{
    br = 2
    x_ax = "Erro Absoluto (%)"
  }
  
  
  p = dat %>% ggplot(aes(x = breaks, y = acc, col = Modelos)) + 
    geom_line(size = 0.9) +
    coord_cartesian(xlim=c(0, x_lim))+
    scale_x_continuous(name=x_ax, breaks = seq(0,x_lim,br)) +
    scale_y_continuous(name="Acurácia", limits=c(0, 1), breaks = seq(0,1,0.1)) +
    theme_classic() +
    theme(panel.background = element_rect(fill = "white", colour = "black"))+
    theme(legend.position = c(0.9, 0.15))+
    theme(legend.background = element_rect(fill="white",
                                           size=0.5, linetype="solid", 
                                           colour ="black")) +
    geom_hline(yintercept = refy, show.legend = T, intercept, linetype = 2) +
    geom_hline(yintercept = 1, show.legend = T, intercept, linetype = 2, col = "red") +
    geom_vline(xintercept = refx, show.legend = T, intercept,linetype = 2)
  
  #return(dat)
  ggsave(filename = paste0("../../part_1/figures/REC_",cenario,"_",target,".png"), plot = p,
         dpi = 1000, width = 6, height = 5)
}

#Function to plot tempo and ur across time

progress = function(target, tec){
  
  pred = list()
  for(cenario in c("cenario_1","cenario_5","cenario_7","cenario_9")){
    pred[[cenario]] = gather_info(cenario, target) 
    
    pred[[cenario]]$tecnica = trimws(pred[[cenario]]$tecnica, which = c("both"))
    
    pred[[cenario]] = pred[[cenario]] %>% filter(tecnica == tec)
    
    pred[[cenario]] = pred[[cenario]] %>% mutate(cenario = cenario)
    
    pred[[cenario]] = pred[[cenario]] %>% 
      gather(variable, value, -(tecnica:cenario)) %>% 
      mutate(concat_coord = paste0(x,y,z,variable,cenario)) 
    
    Timestamp = c(rep(1:(((dim(pred[[cenario]])[1])/45)/2),each = 45),
                  rep(1:(((dim(pred[[cenario]])[1])/45)/2),each = 45))
    
    pred[[cenario]] = pred[[cenario]] %>% 
      mutate(Timestamp = Timestamp)
  }
  
  pred = do.call(rbind,pred) %>% 
    mutate(cenario = case_when(
      cenario == "cenario_1" ~ "Cenário 1c",
      cenario == "cenario_5" ~ "Cenário 2c",
      cenario == "cenario_7" ~ "Cenário 3c",
      cenario == "cenario_9" ~ "Cenário 4c",
      TRUE ~ "erro"
      
    ))
  
  if(target == "temp"){
    #br = 0.5
    y_ax = 'Temperatura (°C)'
  }else{
    #br = 2
    y_ax = 'Umidade relativa (%)'
  }
  
  
  p1  = pred %>% filter(value > 0) %>% 
    ggplot(aes(x = Timestamp,y = value)) + 
    geom_line(aes(group = concat_coord,colour=variable), size = 0.65) + 
    facet_wrap(~cenario, ncol = 2, scales = "free") + 
    xlab('Timestamp') + ylab(y_ax) +
    scale_color_manual(labels = c("Predito", "Real"),values=c(rgb(1,0,0), rgb(0,0,1,0.4))) +
    #facet_grid(tecnica~cenario, scales = "free") + 
    theme_bw() + theme(text = element_text(size = 7)) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
    theme(axis.text.x = element_text(size = 6))+
    theme(axis.text.y = element_text(size=6))+
    theme(axis.title = element_text(size=12))+
    theme(legend.text=element_text(size=10))+
    theme(legend.title = element_text(size = 10))+
    theme(strip.text = element_text(size = 10)) + 
    labs(col="Dados") +
    ggtitle(tec) +
    theme(plot.title = element_text(hjust = 0.5, size = 12)) 
  ggsave(filename = paste0("../../part_1/figures/progress_",tec,"_",
                           target,".png"), plot = p1,
         dpi = 1500, width = 12, height = 6)
  
  
}

# Function to calculate RRMSE for all scenarios tecniques and targets

RRMSE = function(target){
  pred = list()
  for(cenario in c("cenario_1","cenario_5","cenario_7","cenario_9")){
    pred[[cenario]] = gather_info(cenario, target) 
    
    pred[[cenario]]$tecnica = trimws(pred[[cenario]]$tecnica, which = c("both"))
    
    pred[[cenario]] = pred[[cenario]] %>% mutate(cenario = cenario) 
    
  }
  
  df = do.call(bind_rows, pred)
  rrmse_by_coord = df %>% group_by(tecnica, cenario,x,y,z) %>% 
    summarise(media_real = mean(real),rmse = sqrt(mean((predicted - real)^2))) %>% 
    ungroup() %>% 
    group_by(tecnica, cenario,x,y,z) %>% 
    summarise(rrmse = (100/media_real)*rmse)
  
}


# Function to calculate RMSE for each coordinate

RMSE = function(target){
  pred = list()
  for(cenario in c("cenario_1","cenario_5","cenario_7","cenario_9")){
    pred[[cenario]] = gather_info(cenario, target) 
    
    pred[[cenario]]$tecnica = trimws(pred[[cenario]]$tecnica, which = c("both"))
    
    pred[[cenario]] = pred[[cenario]] %>% mutate(cenario = cenario) 
    
  }
  
  df = do.call(bind_rows, pred)
  rmse_by_coord = df %>% group_by(tecnica, cenario,x,y,z) %>% 
    summarise(mae = mean(abs(predicted - real)),rmse = sqrt(mean((predicted - real)^2))) 

}


#Function to plot descriptive data

boxplot_progress = function(cenario, target){
  
  dat = gather_info(cenario, target) %>% 
    filter(tecnica == "brt") %>% #just to select the data once not three times
    mutate(hora = hms(hora)) %>%
    mutate(range_hour = case_when(
      hora > hms("00:00:00") & hora <= hms("03:00:00") ~ "00-03",
      hora > hms("03:00:00") & hora <= hms("06:00:00") ~ "03-06",
      hora > hms("06:00:00") & hora <= hms("09:00:00") ~ "06-09",
      hora > hms("09:00:00") & hora <= hms("12:00:00") ~ "09-12",
      hora > hms("12:00:00") & hora <= hms("15:00:00") ~ "12-15",
      hora > hms("15:00:00") & hora <= hms("18:00:00") ~ "15-18",
      hora > hms("18:00:00") & hora <= hms("21:00:00") ~ "18-21",
      hora > hms("21:00:00") & hora <= hms("24:00:00") ~ "21-24",
      TRUE ~ "ERRO"
    ))
  
  if(target == "temp"){
    #br = 0.5
    y_ax = "Temperatura (°C)"
  }else{
    #br = 2
    y_ax = "Umidade relativa (%)"
  }
  
  p1 =  dat %>% ggplot(aes(x = range_hour, y = real)) +
    stat_boxplot(geom ='errorbar') +
    geom_boxplot() +
    xlab("Intervalo de horas") +
    scale_y_continuous(name=y_ax) +
    theme_classic() +
    theme(panel.background = element_rect(fill="white",
                                          size=0.5, linetype="solid", 
                                          colour ="black")) 
  
  ggsave(filename = paste0("../../part_1/figures/boxplot_",cenario,"_",target,".png"), plot = p1,
         dpi = 1000, width = 10, height = 5)
  
}

sd_progress = function(target){
  
  dat = list()
  cenario = c("cenario_1","cenario_5","cenario_7","cenario_9")
  for(cenario in cenario){
    dat[[cenario]] = gather_info(cenario, target) %>% 
      filter(tecnica == "brt", real > 0) %>% #just to select the data once not three times
      mutate(datas = paste0(data," ",hora)) 
    dates = round_date(ymd_hms(dat[[cenario]]$datas),unit = "1 hour")
    dat[[cenario]] = dat[[cenario]] %>% 
      mutate(datas = dates) %>% 
      group_by(datas) %>%
      summarise(sd_real = (sd(real)/mean(real))) 
    
    dat[[cenario]] = dat[[cenario]] %>% 
      mutate(hora = hour(datas), minuto = minute(datas), segundo = second(datas)) %>%
      mutate(range_hour = paste(hora,minuto,segundo, sep = ":0")) %>% 
      mutate(cenario = cenario) %>% 
      group_by(range_hour,hora, cenario) %>%
      summarise(sd_real = mean(sd_real)) %>% 
      arrange(hora) %>% as.data.frame() %>% 
      mutate(range_hour = ifelse(str_extract(range_hour,"^\\d*") %in% seq(0,9,1), 
                                 paste0("0",range_hour),range_hour))
  }
  
  df = do.call(bind_rows, dat)
  
  if(target == "temp"){
    #br = 0.5
    y_ax = "Homogeneidade"
  }else{
    #br = 2
    y_ax = "Homogeneidade"
  }
  
  (p1 =  df %>%
      ggplot(aes(x = range_hour, y = sd_real, col = cenario)) +
      geom_line(aes(group = cenario)) +
      xlab("Intervalo de horas") +
      scale_y_continuous(name=y_ax) +
      theme_classic() +
      theme(axis.text.x = element_text(size = 8,angle = 30))+
      theme(axis.title.x = element_text(margin = margin(t = 8)))+
      theme(axis.title.y = element_text(margin = margin(r = 8))) +
      theme(panel.background = element_rect(fill="white",
                                            size=0.5, linetype="solid", 
                                            colour ="black"))) 
  
  ggsave(filename = paste0("../../part_1/figures/sd_",target,".png"), plot = p1,
         dpi = 2000, width = 10, height = 5)
  
}

# Execution ---------------------------------------------------------------

alpha = data.frame(expand.grid(
  cenario = c("cenario_1","cenario_5","cenario_7","cenario_9"),
  target = c("temp","ur"),
  tecnica = c("brt","svm","rf"))
) %>% 
  mutate_if(is.factor, as.character)
alpha = split(alpha,list(alpha$cenario,alpha$target,alpha$tecnica), drop=TRUE)

rec_list = lapply(alpha, function(x) Rec_plot(cenario = x[,"cenario"], target = x[,"target"]))
progress_list = lapply(alpha, function(x) progress(target = x[,"target"],
                                                   tec = x[,"tecnica"]))
boxplot_list = lapply(alpha, function(x) boxplot_progress(cenario = x[,"cenario"], target = x[,"target"]))
sd_list = lapply(alpha, function(x) sd_progress( target = x[,"target"]))
