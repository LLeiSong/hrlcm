# Title     : Script to build LC guess model
# Objective : To tune and train random forest 
#             land cover guesser.
# Created by: Lei Song
# Created on: 02/24/21

#######################
##  Step 1: Setting  ##
#######################
message('Step1: Setting')

## Load packages
library(here)
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
message('Step 2: Load training')

## Get training
load(file.path(
    here('data/tanzania'), 'training.rda'))
training <- training %>% 
    # wetland is really not accurate. 
    # so we give up this class.
    filter(landcover != 5) %>% 
    arrange(landcover) %>% 
    mutate(landcover = as.factor(landcover))

##############################
##  Step 3: Tune the model  ##
##############################
message('Step 3: Tune the model')

## Remove NAs and get subset of training to tune
set.seed(123)
training <- training %>% na.omit() %>% 
    sample_frac(size = 0.1)

## Importance of features
message('--Feature importance')
### We want to reduce the number of S1 to mitigate
### the impacts of low resolution.
set.seed(785)
forest_vip <- rand_forest() %>%
    set_engine("ranger", 
               importance = "impurity_corrected") %>%
    set_mode("classification") %>% 
    fit(landcover ~ ., data = training)
save(forest_vip, 
     file = here('data/tanzania/forest_vip.rda'))

## Plot
vip(forest_vip, 
    num_features = forest_vip$fit$num.independent.variables,
    aesthetics = list(color = "transparent", 
                      fill = "red")) +
    theme_light()

## Define the cross validation folds
message('--Cross validation')

set.seed(123)
training_split <- initial_split(
    training, prop = 0.80)
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
save(tune_res, 
     file = here('data/tanzania/tune_res.rda'))

## Check the results
tune_res %>%
    collect_metrics() %>%
    filter(.metric == "accuracy") %>%
    mutate(min_n = factor(min_n)) %>%
    ggplot(aes(mtry, mean, color = min_n)) +
    geom_line() + geom_point() +
    scale_color_brewer(palette = "Dark2") +
    labs(y = "Accuracy") +
    theme_light()
    
## Best parameters
best_acc <- select_best(tune_res, "roc_auc")
final_rf <- finalize_model(
    forest_tune,
    best_acc
)

### Testing
message('--Test')

final_wf <- workflow() %>%
    add_model(final_rf) %>% 
    add_formula(landcover ~ .)

final_res <- final_wf %>%
    last_fit(training_split)
save(final_res, 
     file = here('data/tanzania/final_res.rda'))

#### Evaluation metrics
final_res %>%
    collect_metrics()

cm <- final_res$.predictions[[1]] %>% 
    data.frame() %>% 
    mutate(Prediction = .pred_class,
           Truth = landcover) %>% 
    dplyr::select(Truth, Prediction) %>% 
    conf_mat(truth = Truth, estimate = Prediction)
autoplot(cm, type = 'heatmap')
# autoplot(cm, type = "mosaic")

# Calculate each class accuracy
producer_sum <- colSums(cm$table)
user_sum <- rowSums(cm$table)
producer_accuracy <- sapply(1:7, function(n){
    cm$table[n, n] / producer_sum[n] * 100
}) %>% units::set_units('%')
user_accuracy <- sapply(1:7, function(n){
    cm$table[n, n] / user_sum[n] * 100
}) %>% units::set_units('%')

producer_accuracy
user_accuracy

###############################
##  Step 4: Train the model  ##
###############################
message('Step 4: Train the model')

## Train the model using best parameters
## NOTE: we didn't collect_prediction because
## we just use the subset of the training to tune.

### Clean working environment
rm(lc_test, lc_train, training, training_folds,
   training_split); gc()

### Reload training
load(file.path(here('data/tanzania'), 'training.rda'))

# Or use all features
training <- training %>%
    filter(landcover != 5) %>%
    arrange(landcover) %>%
    mutate(landcover = as.factor(landcover))

### Train the final guess model
guess_rf_md <- final_rf %>% 
    set_engine("ranger", num.threads = 11) %>% 
    fit(landcover ~ ., training)
save(guess_rf_md, 
     file = here('data/tanzania/guess_rf_md.rda'))

############################################
##  Step 5: Predict one tile for example  ##
############################################
message('Step 5: Predict one tile for example')

library(terra)
## Read image stack
fnames <- list.files(
    '/Volumes/elephant/pred_stack',
    full.names = T)
set.seed(123)
img_path <- fnames[sample(length(fnames), 1)]

# Or use all features
imgs <- rast(img_path)

## Predict scores and classes
scores <- predict(imgs, guess_rf_md, type = "prob")
pred <- argmax(values(scores))
classes <- scores[[1]]
values(classes) <- pred
writeRaster(scores, here('data/tanzania/scores.tif'))
writeRaster(classes, here('data/tanzania/classed.tif'))
