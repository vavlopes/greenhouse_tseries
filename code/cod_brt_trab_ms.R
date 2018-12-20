library(mlr)
library(tidyverse)
library(parallel)
library(parallelMap)

rm(list = ls())

years = c("2014","2013","2012","2015")

#criando dat para definir os conjuntos de iteracao
dat_iter = data_frame(a = rep(1:3, each = 4),
                      years = rep(years,3))
dat_iter$concat = paste(dat_iter$a, dat_iter$years, sep="_")

#levantando os results ja obtidos
nomes = list.files("../results/brt/",full.names = TRUE)
if(length(nomes) == 0){
  dat_iter = dat_iter
} else{
  tst = regmatches(nomes, regexpr("[0-9].*[0-9]", nomes))
  tst2 = gsub('_ano', '', tst)
  
  #subtraindo tst2 de dat_iter
  not = match(dat_iter$concat,tst2)
  not = not[is.na(not) == FALSE]
  dat_iter = dat_iter[-not,]
}

for(x in dat_iter$concat){ 
  
  a = as.numeric(substr(x, start = 1, stop = 1))
  i = as.numeric(substr(x, start = 3, stop = 7))
  
  #para um dado ano que quero testar, separo para treino todos os anos diferentes deste
  # Lembrar: para treino sempre pego dados do conjunto original (dados2017)
  # para teste posso ter os dados simulados do Tobias  
  dat_train = read.csv("../../data/data_1.csv") %>% filter(format(as.Date(col), "%Y") != i) #para extrair on inds de train
  dat_test = read.csv(paste0("../../data/data_",a,".csv")) %>% filter(format(as.Date(col), "%Y") == i)
  dat = rbind(dat_train,dat_test)
  clust = as.factor(dat$cluster)
  
  #Separo as linhas de treino e teste, de acordo com o ano de teste  
  train_lines = which(format(as.Date(dat$col), "%Y") != i)
  test_lines = which(format(as.Date(dat$col), "%Y") == i)
  
  #Tirando as colunas que afetarao a modelagem. Massaseca tem correlacao direta co tch, afeta
  # o resultado se deixada no conjunto
  dat = dat[,-grep('^fst$', colnames(dat))]
  dat = dat[,-grep('^col$', colnames(dat))]
  dat = dat[,-grep('^cluster$', colnames(dat))]
  dat = dat[,-grep('^tch$', colnames(dat))]
  
  # cria uma tarefa de regressao para modelar tch (blocking clust mantem as instancias com mesmo
  # cluster "andando" juntas durante toda a execução)
  regr_task = makeRegrTask(id = 'brt', data = dat, target = 'massaSeca', blocking = clust)
  # especifica seed para particionar o conjunto de dados
  set.seed(1)
  rval = makeFixedHoldoutInstance(train.inds = train_lines, test.inds = test_lines, 
                                  size = nrow(dat))
  rmod = makeResampleDesc('CV', iters=5, predict='both')
  
  # especifica que o K deve ser variado de 1 a 20
  parameters = makeParamSet(
    makeDiscreteParam("n.trees", seq(100,1000, by=100)),
    makeDiscreteParam("interaction.depth",seq(1,5,by=1)),
    makeNumericParam("shrinkage",lower = 0.001, upper = 0.2)
  )
  
  # especifica que usaremos uma busca aleatoria
  ctrl = makeTuneControlRandom(maxit = 50L)
  
  parallelStart(mode = 'multicore', cpus = 6, level = 'mlr.tuneParams')
  
  # cria um learner de regressao com gbm, que faz o preprocessing de criar variaveis dummy
  base_learner = makeDummyFeaturesWrapper("regr.gbm")
  
  # considera agora que o learner vai ser o melhor resultado de um procedimento de tunning com CV
  lrn = makeTuneWrapper(base_learner, resampling = rmod, par.set = parameters, control = ctrl, 
                        measures = mae)
  
  # avalia o modelo resultante do tunning com cross-validation (rmod) usando o conjunto de validacao (rval) 
  r = resample(lrn, regr_task, resampling = rval, extract = getTuneResult, show.info = TRUE, 
               models=TRUE, measures = mae) 
  
  parallelStop()
  
  dat_pars = generateHyperParsEffectData(r,partial.dep = TRUE)
  
  write.table(r$measures.test,paste0("../results/brt/mae_",a,"_ano_",i,".txt"))
  resu = r$pred$data
  fst = dat_test[,"fst"]
  write.table(cbind(fst,resu),paste0("../ypred/brt/ypred_",a,"_ano_",i,".txt"))
  write.table(dat_pars$data,paste0("../cv/brt/cv_",a,"_ano_",i,".txt"))
  #plot(ypred$data$truth, ypred$data$response)
}
