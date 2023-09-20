library(ISLR)
library(dplyr)
library(tidyr)
library(skimr)
library(lubridate)
library(frequency)
# Gráficos
# ==============================================================================
library(ggplot2)
library(ggpubr)

# Preprocesado y modelado
# ==============================================================================
library(tidymodels)
library(ranger)
library(doParallel)

library(yardstick)
library(vip)
library(caret)

#MODELO RANDOM FOREST

modelorm <- rand_forest(
  mode  = "classification",
  mtry  = tune(),
  trees = tune()
) %>%
  set_engine(
    engine     = "ranger",
    max.depth  = tune(),
    importance = "impurity",
    seed       = 123
  )


transformer <- recipe(
  formula = ECV ~ .,
  data    =  entrenamientos
)

set.seed(1234)
cv_folds <- vfold_cv(
  data    = entrenamientos,
  v       = 5,
  strata  = ECV
)

workflow_modelado <- workflow() %>%
  add_recipe(transformer) %>%
  add_model(modelorm)


hiperpar_grid <- expand_grid(
  'trees'     = c(50, 100, 500, 1000, 5000),
  'mtry'      = c(3, ncol(entrenamientos)-1),
  'max.depth' = c(1, 3, 10, 20)
)


cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

grid_fit <- tune_grid(
  object    = workflow_modelado,
  resamples = cv_folds,
  metrics   = metric_set(roc_auc),
  grid      = hiperpar_grid
)

stopCluster(cl)

show_best(grid_fit, metric = "roc_auc", n =10)

mejores_hiperpar <- select_best(grid_fit, metric = "roc_auc")

modelorm_f <- rand_forest(
  mode  = "classification"
) %>%
  set_engine(
    engine     = "ranger",
    importance = "impurity",
    seed       = 123
  )

modelorm_f <- modelorm_f %>% finalize_model(mejores_hiperpar)
modelorm_f <- modelorm_f %>% fit(ECV ~., data = entrenamientos)
importanciarm<-varImp(modelorm_f)

#XGBoost

xgb_spec <- boost_tree(
  trees = 1000, 
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(),                     
  sample_size = tune(), mtry = tune(),         
  learn_rate = tune(),                         
) %>% 
  set_engine("xgboost", importance = "impurity") %>% 
  set_mode("classification")

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), entrenamientos),
  learn_rate(),
  size = 30
)


xgb_wf <- workflow() %>%
  add_formula(ECV ~ .) %>%
  add_model(xgb_spec)

set.seed(123)
vb_folds <- vfold_cv(entrenamientos, strata = ECV)

doParallel::registerDoParallel()

set.seed(234)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = vb_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE)
)


show_best(xgb_res, "roc_auc")
best_auc <- select_best(xgb_res, "roc_auc")

set.seed(123)
modeloxg_f <- boost_tree(mode = "classification")%>%
  set_engine(
    engine     = "xgboost",
    importance = "impurity",
    eval_metric="auc"
  )

modeloxg_f <- modeloxg_f %>% finalize_model(best_auc)
modeloxg_f <- modeloxg_f %>% fit(ECV ~., data = entrenamientos)
importanciaxg<-varImp(modeloxg_f)
#NB 
library(caret)
modelo_nb <- train(x=as.matrix(entrenamientos[,-1]),y=factor(entrenamientos$ECV, levels = c("0", "1"), labels = c("No", "Si")),
                   method = "nb",
                   tuneGrid = hiperparametros,
                   metric = "Accuracy",
                   trControl = trainControl(method = "cv", number = 5,summaryFunction = twoClassSummary, classProbs = TRUE))
modelo_nb
importancianb<-varImp(modelo_nb)
