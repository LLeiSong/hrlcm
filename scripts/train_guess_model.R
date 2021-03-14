# Title     : Script to build LC guess model
# Objective : To tune and train random forest 
#             land cover guesser.
# Created by: Lei Song
# Created on: 02/24/21

#######################
##  Step 1: Setting  ##
#######################
## Load packages
library(dplyr)
library(ramify)
library(ranger)
library(parsnip)
library(tidymodels)
library(vip)
library(dials)
library(tune)
library(ggplot2)

#############################
##  Step 2: Load training  ##
#############################
## Get training
load(file.path('data/north', 'training.rda'))
training <- training %>% 
    # wetland is really not accurate. so we give up this class.
    filter(landcover != 5) %>% 
    arrange(landcover) %>% 
    mutate(landcover = as.factor(landcover))

##############################
##  Step 3: Tune the model  ##
##############################
## Remove NAs and get subset of training to tune
set.seed(123)
training <- training %>% na.omit() %>% 
    sample_frac(size = 0.1)

## Importance of features
### We want to reduce the number of S1 to mitigate
### the impacts of low resolution.
set.seed(785)
forest_vip <- rand_forest() %>%
    set_engine("ranger", 
               importance = "impurity_corrected") %>%
    set_mode("classification") %>% 
    fit(landcover ~ ., data = training)
save(forest_vip, file = 'data/north/forest_vip.rda')

## Plot
vip(forest_vip, num_features = ncol(training) - 1,
    aesthetics = list(color = "transparent", 
                      fill = "red")) +
    theme_light()

## Based on the figure above, we decide to get rid of 
## vv (3, 5, 6) and vh(3, 5, 6) variables. Because they
## are the least important features, and they have coarse
## spatial resolution.

## Subset the features based on importance
training <- training %>% 
    dplyr::select(-c(paste0('vv', c(3, 5, 6)),
                     paste0('vh', c(3, 5, 6))))

## Define the cross validation folds
set.seed(123)
training_split <- initial_split(training, prop = 0.80)
lc_train <- training(training_split)
lc_test <- testing(training_split)

## Set folds for cross validation
set.seed(234)
training_folds <- vfold_cv(lc_train, v = 5)

## Tune parameters of RF
### Define tune model
forest_tune <- rand_forest(
    mtry = tune(),
    trees = 1000,
    min_n = tune()) %>%
    set_mode("classification") %>%
    set_engine("ranger")

### Connect to workflow
tune_wf <- workflow() %>%
    add_model(forest_tune) %>% 
    add_formula(landcover ~ .)

### Define search grid
rf_grid <- grid_regular(
    mtry(range = c(4, 22)),
    min_n(range = c(2, 20)),
    levels = 6
)

### Tuning
doParallel::registerDoParallel(12)
tune_res <- tune_grid(
    tune_wf,
    resamples = training_folds,
    grid = rf_grid,
    control = control_grid(verbose = TRUE)
)
save(tune_res, file = 'data/north/tune_res.rda')

## Check the results
tune_res %>%
    collect_metrics() %>%
    filter(.metric == "accuracy") %>%
    mutate(min_n = factor(min_n)) %>%
    ggplot(aes(mtry, mean, color = min_n)) +
    geom_line() + geom_point() +
    scale_color_brewer(palette="Dark2") +
    labs(y = "Accuracy") +
    theme_light()
    
## Best parameters
best_acc <- select_best(tune_res, "roc_auc")
final_rf <- finalize_model(
    forest_tune,
    best_acc
)

### Testing
final_wf <- workflow() %>%
    add_model(final_rf) %>% 
    add_formula(landcover ~ .)

final_res <- final_wf %>%
    last_fit(training_split)
save(final_res, file = 'data/north/final_res.rda')

final_res %>%
    collect_metrics()

###############################
##  Step 4: Train the model  ##
###############################
## Train the model using best parameters
## NOTE: we didn't collect_prediction because
## we just use the subset of the training to tune.

### Clean working environment
rm(lc_test, lc_train, training, training_folds,
   training_split); gc()

### Reload training
load(file.path('data/north', 'training.rda'))
training <- training %>%
    filter(landcover != 5) %>%
    arrange(landcover) %>% 
    mutate(landcover = as.factor(landcover)) %>% 
    dplyr::select(-c(paste0('vv', c(3, 5, 6)),
                     paste0('vh', c(3, 5, 6))))

### Train the final guess model
guess_rf_md <- final_rf %>% 
    set_engine("ranger", num.threads = 6) %>% 
    fit(landcover ~ ., training)
save(guess_rf_md, file = 'data/north/guess_rf_md.rda')

############################################
##  Step 5: Predict one tile for example  ##
############################################
library(terra)
## Read image stack
fnames <- list.files('/Volumes/elephant/pred_stack',
                     full.names = T)
set.seed(67)
img_path <- fnames[sample(length(fnames), 1)]
imgs <- rast(img_path) %>% 
    subset(setdiff(names(.), 
                   c(c(paste0('vv', c(3, 5, 6)),
                       paste0('vh', c(3, 5, 6))),
                     'rivers_dist', 'buildings_dist', 
                     'roads_dist')))

## Predict scores and classes
scores <- predict(imgs, guess_rf_md, type = "prob")
pred <- argmax(values(scores))
classes <- scores[[1]]
values(classes) <- pred
writeRaster(scores, 'data/north/scores.tif')
writeRaster(classes, 'data/north/classed.tif')
