# Combining non-probability sample with probability sample using random forest
# Author: StatsEye 

### Prepare R environment

options(scipen=999)
rm(list = ls())

.libPaths(".../R_library")

# Install required packages

install.packages("haven")
install.packages("here")
install.packages("magrittr")
install.packages("tidyverse")
install.packages("randomForest")
install.packages("caret")
install.packages("ranger")
install.packages("e1071")
install.packages("dplyr")

# Load required packages

library(haven)
library(here)
library(magrittr)
library(tidyverse)
library(randomForest)
library(caret)
library(ranger)
library(e1071)
library(dplyr)

# Read data

data <- read_sav(here("folder", "dataset with two samples.sav"))

### Data management

# Recode dependent variable

data %>%
count(survey)

data %<>%
mutate(survey_prob = if_else(survey == 1, 
true = 0,
false = 1) %>%
as.factor())

table(data$survey_prob)

# Rename variables

names(data) <- tolower(names(data))

### Estimate the accuracy of logistic regression model (benchmark) 

# Define the model formula

rf_formula <- tolower("survey_prob ~ region + age7sex + relationship + anychild + economic + tenure + internetuse + education + ethnic") %>% as.formula()
rf_formula

# Run logit model

logit_m <- glm(rf_formula,
data = data,
family = 'binomial',
# apply the non-response weight for probability sample, equals 1 for non-probability sample
weights = data$weight_prob)

logit_m

# Test predictions on full dataset and calculate accuracy
predicted_propensities_logit <- predict.glm(logit_m, newdata=data, type="response")
predicted_response_logit <- ifelse(predicted_propensities_logit > 0.5, 1, 0)
accuracy <- table(predicted_response_logit, data$survey_prob)
sum(diag(accuracy))/sum(accuracy)
# accuracy of a logit = 0.726

### Estimate accuracy of a random forest response model and compare with logistic regression

# RF parameters to tune:
# * mtry - number of variables randomly samples as candidates at each split;
# * ntree = number of trees to grow. 500 is default
# * min_sample_leaf* - leaf is an end node of a decision tree - a smaller leaf makes the model more prone to capturing noise in the data. 
# I will set the min. sample leaf to 100 to avoid high variability of a final weight 
# (this increases design effect and impacts on effective sample size for analysis). 
# I'll search for optimal parameters, i.e. set of parameters that maximize prediction accuracy. 

# Use train() from the caret package. Train can be used to tune models by
# picking the complexity parameters that are associated with the optimal
# resampling statistics. For particular model, a grid of parameters (if any) is
# created and the model is trained on slightly different data for each candidate
# combination of tuning parameters. Across each data set, the performance of
# held-out samples is calculated and the mean and standard deviation is
# summarized for each combination. The combination with the optimal resampling
# statistic is chosen as the final model and the entire training set is used to
# fit a final model. I will use train() function just to show how a selection of
# a model can be done if we were interested in mazimizing out-of-bag accuracy. In
# this case, we want to maximizing the accuracy on the full dataset. For
# estimation of variable importance I selected the impurity_corrected - an
# importance measure that is unbiased in terms of the number of categories and
# category frequencies.

# Define grid for testing different parameters
tgrid <- expand.grid(.mtry = 3:9, .splitrule = c("gini", "extratrees"), .min.node.size = c(100, 120, 150))
tgrid

# Tune RF models
set.seed(1234)
RF_tune <- caret::train(survey_prob ~ region + age7sex + relationship + anychild + economic + tenure + internetuse + education + ethnic,
data = data,
method = "ranger",
trControl = trainControl(method="oob", number = 5, verboseIter = T),
tuneGrid = tgrid,
num.trees = 1000,
importance = "impurity_corrected", 
case.weights = data$weight_prob)

RF_tune

# parameters and confusion matrix of the best model
RF_tune$finalModel
# prediction.error: overall out of bag prediction error. 
# For classification this is the fraction of missclassified samples. 

# max accuracy of rf model (out of bag)
RF_tune$results$Accuracy  %>% max()
# accuracy of a logit = 0.726. Hence, RF can fit the data better. 

RF_tune$results$Accuracy  %>% summary()
# accuracy of any of the random forest models can be better than of the logit model

# variables importance measured with the impurity_corrected
varImp(RF_tune)

### Find the best probability estimation model (random forest, ranger package)

# In order to compute probability of group membership which can be later used to
# create a weight, we need to use probability estimation. I will use the
# ranger package to grow a probability forest (probability = T). 
# Read Malley JD, Kruppa J, Dasgupta A, Malley KG, Ziegler A (2012). "Probability Machines:
# Consistent Probability Estimation Using Nonparametric Learning Machines."
# Methods of Information in Medicine, 51(1), 74. doi:10.3414/me00-01-0052.
#
# Here, the node impurity is used for splitting, as in classification forests.
# Predictions are class probabilities for each sample. In contrast to other
# implementations, each tree returns a probability estimate and these estimates
# are averaged for the forest probability estimate.
#
# I will manually create a search grid to look for a model that maximizes the
# accuracy on the full dataset.

# Define grid for testing models with different parameters
hyper_grid <- expand.grid(mtry = 3:9,
min.node.size = c(100, 120, 150),
num.trees = 1000,
accuracy = 0)

system.time(
for(i in 1:nrow(hyper_grid)) {
# train model
rf <- ranger(
formula        = rf_formula,
data           = data,
probability = T,
num.trees      = hyper_grid$num.trees[i],
mtry           = hyper_grid$mtry[i],
min.node.size  = hyper_grid$min.node.size[i],
importance = 'impurity_corrected', 
case.weights = data$weight_prob, 
seed = 1234)

# Retrieve propensities and calculate accuracy
predicted_rf <- predict(rf, data = data)
predicted_propensities_rf <- predicted_rf$predictions[,2]
predicted_response_rf <- ifelse( predicted_propensities_rf > 0.5, 1, 0)
accuracy <- table(predicted_response_rf, data$survey_prob)

# add accuracy to grid
hyper_grid$accuracy[i] <- sum(diag(accuracy))/sum(accuracy)
})

# Number of models tested
nrow(hyper_grid)

hyper_grid

# Optimal parameters set maximizing accuracy
position = which.max(hyper_grid$accuracy)
head(hyper_grid[order(hyper_grid$accuracy, decreasing = TRUE),],5)
# highest accuracy = 0.758 for mtry=5 and min.node.size = 100. 

### Compute probability of group membership using the best model

# Fit the best model
rf.best <- ranger(formula = rf_formula,
data = data,
probability = T,
num.trees = 1000,
importance = "impurity_corrected",
min.node.size = hyper_grid$min.node.size[position],
mtry = hyper_grid$mtry[position],
case.weights = data$weight_prob, 
seed = 1234)

rf.best

# Retrieve propensities and create weight
predicted_rf.best <- predict(rf.best, data = data)
head(predicted_rf.best$predictions)
predicted_propensities_rf.best <- predicted_rf.best$predictions[,2] 

# Compare to predicted propensities from a logit model
summary(predicted_propensities_logit)
summary(predicted_propensities_rf.best)

# Pearson correlation coeff
cor(predicted_propensities_logit, predicted_propensities_rf.best)

### Calculate weight
rf_weight <- 1/predicted_propensities_rf.best
summary(rf_weight)

# The weight will be applied for each case in the non-probability sample. 
# It should  align its profile with the probabilistic sample's profile on the
# variables used in the model.

### Export weight and propensities as .csv

data_frame(id = data$id,
           predicted_propensities = predicted_propensities_rf.best,
           rf_weight = rf_weight) %>%
  write_csv(here("folder", "weights", "propensities_and_weights.csv"))

### Extra - Brier score for an optimal model for probabilistic forecasts

# If we were to select an optimal model for probabilistic forecasts (aimed to be
# used on different data), the Brier score can be optimized
# (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5074325/). The Brier score,
# named for Glenn Brier, calculates the mean squared error between predicted
# probabilities and the expected values. The error score is always between 0.0
# and 1.0, where a model with perfect skill has a score of 0.0. Hence, a small
# Brier Score indicates high prediction accuracy. Below I am presenting a search
# grid that allows to look for a model that optimizes the Brier score. The
# tuneRanger package could be used for this, but I could not properly load the
# library. https://arxiv.org/pdf/1804.03515.pdf
# https://rdrr.io/cran/tuneRanger/man/tuneRanger.html


hyper_grid <- expand.grid(mtry = 3:9,
                          min.node.size = c(50,100,150),
                          num.trees = 1000,
                          OOB_BrierScore = 0)

system.time(
  for(i in 1:nrow(hyper_grid)) {
    # train model
    rf <- ranger(
      formula        = rf_formula,
      data           = data,
      probability = T,
      num.trees      = hyper_grid$num.trees[i],
      mtry           = hyper_grid$mtry[i],
      min.node.size  = hyper_grid$min.node.size[i],
      importance = 'impurity_corrected', 
      case.weights = data$weight_prob, 
      seed = 1234)
    
    # add OOB error to grid
    hyper_grid$OOB_BrierScore[i] <- rf$prediction.error
  })

nrow(hyper_grid)

position = which.min(hyper_grid$OOB_BrierScore)
head(hyper_grid[order(hyper_grid$OOB_BrierScore),],5)

