# Decision tree without weights

library(tidyverse)
library(dplyr)
library(skimr)
library(janitor)
library(ggplot2)
library(rsample)
library(caret)
library(ROCR)
library(vip)
library(pdp)

# Importing the dataset
library(readr)
vh_data14 <- read_csv("vh_data14.csv")

##Encoding Variables
vh_data14$race = factor(vh_data14$race, levels = c("1", "2", "3", "4"), ordered = F)
table(vh_data14$race)                    

vh_data14<-vh_data14 %>%
  rename(biden = president_approval, 
         trump = trump_approval_retrospective,
         perceived_personal_risk = perceived_personal_riskq297_4)


vh_data14$biden = factor(vh_data14$biden, levels = c("No","Yes"))
vh_data14$trump = factor(vh_data14$trump, levels = c("No", "Yes"))

vh_data14$party_id = factor(
  vh_data14$party_id, 
  levels = c("Republican", "Democrat", "Independent", "Libertarian", "Other party"))

## Encoding the target feature as factor
vh_data14$vaccine_hesitant = factor(vh_data14$vaccine_hesitant, levels = c(0, 1))
table(factor(vh_data14$vaccine_hesitant))

vh_data14 <- vh_data14[is.na(vh_data14$vaccine_hesitant)==F,]

vh_data14 <- vh_data14 %>%
  select(-round, - respondent_id, -response_round_2, -obs, -inv_p)

vh_data14 <- na.omit(vh_data14)


##Decision Tree without weights

#Data Partition
library(tidymodels)
set.seed(2345)
dataset_split1 <- initial_split (vh_data14, prop=0.75, strata = vaccine_hesitant)
training_set1<- training(dataset_split1)
test_set1 <- testing (dataset_split1)


# Model specification
library(tidymodels)
tree_spec <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("classification")

# A model specification is fit using a formula to training data
my_tree1 <- tree_spec %>%
  fit(formula = vaccine_hesitant ~ ., 
      data = training_set1)

my_tree1


#Plotting the tree
library(rpart)

my_tree_plot1 <- rpart(vaccine_hesitant ~ ., 
                       data = training_set1,
                       method = "anova")  

library(rpart.plot)
rpart.plot(my_tree_plot1, type = 1, shadow.col = "darkgray") 

##Pruning
plotcp(my_tree_plot1)

my_tree_plot1 <- rpart(vaccine_hesitant ~ ., 
                       data = training_set1,
                       method = "anova", 
                       cp=.024)  
rpart.plot(my_tree_plot1, type = 1, shadow.col = "darkgray") 

##Prediction on training set

prediction <- predict(my_tree1, new_data = training_set1, type = "class")
head (prediction, 3)

table (prediction)

###Combine predictions and truth values
pred_combined <- prediction %>%
  mutate(true_class = training_set1$vaccine_hesitant)
head(pred_combined,3)

###Confusion Matrix - Accuracy - Sensitivity - Specificity
conf_mat(data = pred_combined,
         estimate = .pred_class,
         truth = true_class)

accuracy(data = pred_combined,
         estimate = .pred_class,
         truth = true_class)

sens(data = pred_combined,
     estimate = .pred_class,
     truth = true_class)

spec(data = pred_combined,
     estimate = .pred_class,
     truth = true_class)

## Calculate the ROC curve for all thresholds
prediction2 <- predict(my_tree1, new_data = training_set1, type = "prob")%>%
  bind_cols(training_set1)
head (prediction2, 3)

##ROC
roc <- roc_curve(prediction2,
                 estimate = .pred_0,
                 truth = vaccine_hesitant)
## Plot the ROC curve
autoplot(roc)

##Calculate area under curve
roc_auc(prediction2,
        estimate = .pred_0,
        truth = vaccine_hesitant)

#Cross Validation

set.seed(2345)
my_folds <- vfold_cv(training_set1, v = 10)

fits_cv <- fit_resamples(tree_spec,
                         vaccine_hesitant ~ .,
                         resamples = my_folds,
                         metrics = metric_set(roc_auc, accuracy))

fits_cv

## Collect metrics of all model runs
all_metrics <- collect_metrics(fits_cv,
                               summarize = F)
print(all_metrics)

#Collect and summarize accuracy of all model runs
collect_metrics(fits_cv)

#Tuning

doParallel::registerDoParallel()

spec_untuned <- decision_tree(
  min_n = tune(),
  tree_depth = tune(),
  cost_complexity = tune())%>%
  set_engine("rpart") %>%
  set_mode("classification")

tree_grid <- grid_regular(
  parameters(spec_untuned),
  levels = 4)

tune_results <- tune_grid(
  spec_untuned,
  vaccine_hesitant ~ .,
  resamples = my_folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc, accuracy))

show_best(tune_results, "roc_auc" )
show_best(tune_results, "accuracy")

##Visualizing tuning results
autoplot(tune_results)

# Select the best performing parameters
final_params <- select_best(tune_results, metric = "roc_auc")
final_params1 <- select_best(tune_results, metric = "accuracy")

final_params
final_params1

# Plug them into the specification
best_spec <- finalize_model(spec_untuned,
                            final_params)
best_spec

#Finalize the model
Tuned_tree <- decision_tree( min_n = 27, tree_depth = 10, cost_complexity = 1e-10) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")%>% 
  fit(formula = vaccine_hesitant ~ .,
      data = training_set1)

# variable Importance Plot 
vi(Tuned_tree, scale = T)

VIP <- Tuned_tree %>%
  vip(
    geom = "col", #"point"
    #all_permutations = T, jitter = T,
    #num_features = 15L,
    include_type = T, scale = T,
    aesthetics = list(color = "white", fill = "blue", size = 0.8, alpha=.7)
  ) +
  labs (title = "Variable Importance Plot for Decison Tree \n The Most Important 10 Variable"
        )

VIP +
  theme(axis.text = element_text(colour = "brown", size = rel(1.3) #, angle = 45
                                 ),
          title = element_text(colour = "Black", size = rel(1.5)
          ))+
  theme(plot.title = element_text(hjust = 0.5))

#Predictions on the Validation/Testing Set
prediction_unweighted <- predict(Tuned_tree, new_data = test_set1, type = "class")
head (prediction_unweighted, 4)

##Combine predictions and truth values
pred_combined_unweighted <- prediction_unweighted %>%
  mutate(true_class = test_set1$vaccine_hesitant)
head (pred_combined_unweighted, 4)

##Confusion Matrix - Accuracy - Sensitivity - Specificity
conf_mat(data = pred_combined_unweighted,
         estimate = .pred_class,
         truth = true_class)

accuracy(data = pred_combined_unweighted,
         estimate = .pred_class,
         truth = true_class)

sens(data = pred_combined_unweighted,
     estimate = .pred_class,
     truth = true_class)

spec(data = pred_combined_unweighted,
     estimate = .pred_class,
     truth = true_class)

## Calculate the ROC curve for all thresholds
prediction2_unweighted <- predict(Tuned_tree, new_data = test_set1, type = "prob")%>%
  bind_cols(test_set1)
head (prediction2_unweighted,4)

##ROC
roc <- roc_curve(prediction2_unweighted,
                 estimate = .pred_0,
                 truth = vaccine_hesitant)
## Plot the ROC curve
autoplot(roc)

##Calculate area under curve
roc_auc(prediction2_unweighted,
        estimate = .pred_0,
        truth = vaccine_hesitant)

