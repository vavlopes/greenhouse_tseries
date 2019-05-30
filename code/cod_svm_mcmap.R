library(e1071)
library(parallel)
library(dplyr)
library(tidyr)

rm(list=ls())

#Global var
mc = 70
csize = 480

# Defining functions ------------------------------------------------------

#Data pre processing
data_process = function(file,hmlook_back,target){
  
  dat = get(load(file))
  
  #definindo o range de look_back
  range_lback = paste0("prev_",seq(hmlook_back,6), collapse = "|")
  
  #para ordenar os dados por data e hora
  dat = dat %>% select(c(data,hora,medicao, concat_coord, cenario, range_datas),
                       c(target),
                       c(x,y,z),
                       matches(range_lback)) #mantenho as colunas que contem look_back
  
  #criando variaveis para compor o arquivo de resultados
  cenario = unique(dat$cenario)
  
  # mantaining only complete caes (in function of variable look back (done))
  dat = dat[complete.cases(dat),] %>% arrange(data,hora)
  
  #train_lines are those considering the 7 first dates. Test is all the three remaining days
  train_lines = which(dat$data %in% unique(dat$data)[1:7]) #sete primeiras datas
  test_lines = which(dat$data %in% unique(dat$data)[8:10])
  
  #infos for results saving
  hora = dat[test_lines,]$hora
  data = dat[test_lines,]$data
  range_datas = unique(dat$range_datas)
  concat_coord = dat[test_lines,]$concat_coord
  
  #Tirando as colunas que afetarao a modelagem. Ou possuem correlacao ou nao fazem parte do set up
  dat = dat %>% select(-c(data,hora,medicao, concat_coord, cenario, range_datas))
  dat = dat %>% mutate_at(vars("x","y","z"), funs(as.factor))
  
  onehot = model.matrix(~.-1,dat[,c("x","y","z")])
  onehot = onehot %>% as.data.frame()
  
  dat = dat %>% select(-c(x,y,z))
  dat = cbind(dat,onehot)
  
  dat_train = dat[train_lines,]
  #dat_train = dat_train[1:400,] #!!!mudar
  #limpeza das colunas com apenas um valor(length(unique) verifica o numero de valores unicos)
  dat_train = dat_train[,apply(dat_train, 2, function(col) { length(unique(col)) > 1 })]
  
  dat_test = dat[test_lines,]
  dat_test = dat_test[,colnames(dat_train)]
  
  l_geral = list()
  l_geral[['dat']] = dat
  l_geral[['dat_train']] = dat_train
  l_geral[['dat_test']] = dat_test
  l_geral[['hora']] = hora
  l_geral[['data']] = data
  l_geral[['range_datas']] = range_datas
  l_geral[['concat_coord']] = concat_coord
  l_geral[['cenario']] = cenario
  
  return(l_geral)
}

#Function for parallel processing
par_svm = function(target, dat_train, fold_index, i, fold, eps, cost, gamma){
  
  to = Sys.time()
  
  #renaming
  dat_train = dat_train %>% rename("target" = target)
  
  split_train = dat_train[fold_index != fold, ]
  split_test = dat_train[fold_index == fold, ]
  
  m = svm(target ~ ., split_train, 
          cost = cost,
          gamma = gamma, 
          epsilon=eps, cachesize = csize)
  
  mae = mean(abs(predict(m, split_test) - split_test[,"target"]))
  
  print(i)
  t1 = Sys.time()
  print(t1-to)
  print(mae)
  return(mae)
}

Hold_out = function(dat_train,dat_test, target, eps, cost, gamma){
  
  to = Sys.time()
  
  dat_train = dat_train %>% rename("target" = target)
  
  m = svm(target ~ ., dat_train, cost = cost,
          gamma = gamma, epsilon=eps, cachesize = csize)
  
  mae = mean(abs(predict(m, dat_test) - dat_test[,target]))
  
  l_final = list()
  l_final[["mae"]] = mae
  l_final[["ypred"]] = predict(m, dat_test)
  l_final[["yobs"]] = dat_test[,target]
  
  print(mae)
  t1 = Sys.time()
  print(t1-to)
  return(l_final)
}


# Executing ---------------------------------------------------------------

#collectiong data to be modeled
files_tbmodel = list.files('../../data/',
                           full.names = T,
                           pattern = ".RData")

target = c("ur","temp")
hmlook_back = seq(1:6)

#para criar um dataframe com todos os inputs possiveis para a func de forecast com svm
df = data.frame(expand.grid(files_tbmodel = files_tbmodel,
                            target = target,
                            hmlook_back = hmlook_back, stringsAsFactors = F))
#criando lista a partir de df para iterar
lista = split(df,list(df$files_tbmodel,df$target,df$hmlook_back))

for(i in 1:length(lista)){
  
  j = 50
  set.seed(42)
  tune_svm = data.frame(i = 1:(j*5),
                        cost= sample(2^(-5:5),j,TRUE),
                        gamma = sample(2^(-8:3),j,TRUE),
                        eps = sample(seq(0.005,0.5,0.001),j,TRUE),
                        fold = rep(1:5, each=j))
  
  target = lista[[i]][,'target']
  hmlook_back = lista[[i]][,'hmlook_back']
  
  l_geral = data_process(file = lista[[i]][,'files_tbmodel'],
                         hmlook_back = hmlook_back,
                         target = target)
  
  dat = l_geral[['dat']]
  dat_train = l_geral[['dat_train']]
  dat_test = l_geral[['dat_test']]
  hora = l_geral[['hora']]
  data = l_geral[['data']]
  range_datas = l_geral[['range_datas']]
  concat_coord = l_geral[['concat_coord']]
  cenario = l_geral[["cenario"]]
  
  
  fold_index = rep(1:5, each=(nrow(dat_train)/5))
  fold_index = fold_index[1:dim(dat_train)[1]]
  
  res = mcmapply(function(i, f, e, cs, g) par_svm(target, dat_train, fold_index, i, f, e, cs, g),
                 tune_svm$i, tune_svm$fold, tune_svm$eps, tune_svm$cost, tune_svm$gamma,
                 mc.cores = mc, mc.set.seed = FALSE)
  
  tune_svm = tune_svm %>% mutate(mae = res)
  
  best = tune_svm %>% group_by(cost,gamma,eps) %>% 
    summarise(media = mean(mae)) %>% arrange(media)
  
  #Holdout step
  
  l_final = Hold_out(dat_train,dat_test, target, 
                     best[1,"eps"], best[1,"cost"], best[1,"gamma"])
  
  #Formatando o resultado final
  resu = data.frame(target = target,
                    cenario = cenario,
                    hmlook_back = hmlook_back,
                    hora = hora,
                    data = data,
                    range_datas = range_datas,
                    concat_coord = concat_coord,
                    tecnica = "svm",
                    yobs= l_final[["yobs"]],
                    ypred = l_final[["ypred"]])
  
  resu = resu[,c('tecnica','cenario','range_datas','concat_coord','data','hora','target','hmlook_back','yobs','ypred')] 
  write.table(resu,paste0("../../results/svm/ypred/ypred_",target,"_",cenario,"_",hmlook_back,".txt"))
  write.table(l_final[["mae"]],paste0("../../results/svm/mae_final/mae_",target,"_",cenario,"_",hmlook_back,".txt"))
  write.table(best,paste0("../../results/svm/cv/cv_",target,"_",cenario,"_",hmlook_back,".txt"))
  
}





