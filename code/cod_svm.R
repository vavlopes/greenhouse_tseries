library(mlr)
library(tidyverse)
library(parallel)
library(parallelMap)

rm(list = ls())
gc()

make_forecast_svm = function(file, hmlook_back, target){ 
  
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
  concat_coord = dat[test_lines,]$data
  
  #creating blocking column for CV - train_lines(1:5) and test_lines(6)
  blocking_train = rep(1:5, each = ceiling(length(train_lines)/5))
  blocking_train = blocking_train[1:length(train_lines)]
  blocking_test = rep(6, length(test_lines))
  block = as.factor(c(blocking_train,blocking_test))
  
  #Tirando as colunas que afetarao a modelagem. Ou possuem correlacao ou nao fazem parte do set up
  dat = dat %>% select(-c(data,hora,medicao, concat_coord, cenario, range_datas))
  dat = dat %>% mutate_at(vars("x","y","z"), funs(as.factor))
  
  
  regr_task = makeRegrTask(id = 'svm', data = dat, target = target, blocking = block)
  # especifica seed para particionar o conjunto de dados
  set.seed(1)
  rval = makeFixedHoldoutInstance(train.inds = train_lines, test.inds = test_lines, 
                                  size = nrow(dat))
  rmod = makeResampleDesc(method = 'CV', predict='test', 
                          iters = 5)
  
  # especifica que o K deve ser variado de 1 a 20
  parameters = makeParamSet(
    makeDiscreteParam("cost", values = 2^(-2:2)),
    makeDiscreteParam("gamma", values = 2^(-2:2)),
    makeNumericParam("epsilon",lower = 0.05, upper = 0.3)
  )
  
  # especifica que usaremos uma busca aleatoria
  ctrl = makeTuneControlRandom(maxit = 50L)
  
  parallelStart(mode = 'multicore', cpus = 12, level = 'mlr.tuneParams')
  
  # cria um learner de regressao com svm, que faz o preprocessing de criar variaveis dummy
  base_learner = makeDummyFeaturesWrapper("regr.svm")
  
  # considera agora que o learner vai ser o melhor resultado de um procedimento de tunning com CV
  lrn = makeTuneWrapper(base_learner, resampling = rmod, par.set = parameters, control = ctrl, 
                        measures = mae)
  
  # avalia o modelo resultante do tunning com cross-validation (rmod) usando o conjunto de validacao (rval) 
  r = resample(lrn, regr_task, resampling = rval, extract = getTuneResult, show.info = TRUE, 
               models=TRUE, measures = mae) 
  
  parallelStop()
  
  dat_pars = generateHyperParsEffectData(r,partial.dep = TRUE)
  
  write.table(r$measures.test,paste0("../../results/svm/mae_final/mae_",target,"_",cenario,"_",hmlook_back,".txt"))
  resu = r$pred$data
  resu$target = target
  resu$cenario = cenario
  resu$hmlook_back = hmlook_back
  resu$hora = hora
  resu$data = data
  resu$range_datas = range_datas
  resu$concat_coord = concat_coord
  resu$tecnica = "svm"
  resu = resu[,c('tecnica','cenario','range_datas','concat_coord','data','hora','target','hmlook_back','truth','response')] %>% 
    rename(yobs = truth, ypred = response)
  write.table(resu,paste0("../../results/svm/ypred/ypred_",target,"_",cenario,"_",hmlook_back,".txt"))
  write.table(dat_pars$data,paste0("../../results/svm/cv/cv_",target,"_",cenario,"_",hmlook_back,".txt"))
}


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

results = lapply(lista, function(x) make_forecast_svm(file = x[,'files_tbmodel'],
                                                      hmlook_back = x[,'hmlook_back'],
                                                      target = x[,'target']))






