# Importing the dataset
library(readr)
vh_data15 <- read_csv("vh_data15.csv")

#Encoding Variables
vh_data15$Race = factor(vh_data15$Race, levels = c("1", "2", "3", "4"), ordered = F)
table(vh_data15$Race)                    

vh_data15$Biden = factor(vh_data15$Biden, levels = c("No","Yes"))

vh_data15$Trump = factor(vh_data15$Trump, levels = c("No", "Yes"))

vh_data15$Party_ID = factor(vh_data15$Party_ID, levels = c("Republican", "Democrat", "Independent", "Libertarian", "Other party"))


# Encoding the target feature as factor
vh_data15$Vaccine_Hesitant = factor(vh_data15$Vaccine_Hesitant, levels = c(0, 1))
table(factor(vh_data15$Vaccine_Hesitant))

#Removing the NAs
dataset <- vh_data15[is.na(vh_data15$Vaccine_Hesitant)==F,]
dataset <- na.omit(dataset)

#Data Partition
library(tidymodels)
set.seed(1234)
dataset_split <- initial_split (dataset, prop=0.75, strata = Vaccine_Hesitant)
training_set<- training(dataset_split)
test_set <- testing (dataset_split)

#Boosted tree

# Specify the model class
#install.packages("xgboost")
library(xgboost)

boost_spec<-boost_tree()%>%
  set_mode("classification")%>%
  set_engine("xgboost")

model_boost<-boost_spec %>%
  # Set the mode
  set_mode("classification") %>%
  # Set the engine
  set_engine("xgboost")%>%
  fit(formula = Vaccine_Hesitant ~ ., data = training_set)

model_boost

#Cross Validation

##Random seed for reproducibility
set.seed(1234)
## Create 10 folds of the dataset
my_folds <- vfold_cv(training_set, v = 10)

fits_cv <- fit_resamples(boost_spec,
                         Vaccine_Hesitant ~ .,
                         resamples = my_folds,
                         metrics = metric_set(accuracy, roc_auc))

fits_cv

## Collect accuracy and ROC_AUC of all model runs
all_accuracy <- collect_metrics(fits_cv,
                               summarize = FALSE)
print(all_accuracy)

library(ggplot2)
ggplot(all_accuracy, aes(x = .estimate,
                        fill = .metric)) +
  geom_histogram()

#Collect and summarize accuracy of all model runs
collect_metrics(fits_cv)

#Tuning 
doParallel::registerDoParallel()

spec_untuned_boost<- boost_tree(
  trees = 1500,
  learn_rate = tune(),
  tree_depth = tune(),
  sample_size = tune()) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

boost_grid <- grid_regular(
  parameters(spec_untuned_boost),
  levels = 4)

tune_results <- tune_grid(
  spec_untuned_boost,
  Vaccine_Hesitant ~ .,
  resamples = vfold_cv(training_set, v = 5),
  grid = boost_grid,
  metrics = metric_set(accuracy, roc_auc))

tune_results
##Visualizing tuning results
autoplot(tune_results)

# Select the best performing parameters
final_params <- select_best(tune_results, metric = "accuracy")
final_params1 <- select_best(tune_results, metric = "roc_auc")

final_params
final_params1

# Plug them into the specification
best_spec <- finalize_model(spec_untuned_boost,
                            final_params)
best_spec

#Tuned Boosting
tuned_boost<- boost_tree(
  trees = 800,
  learn_rate = 0.1,
  tree_depth = 1,
  sample_size = 1) %>%
  set_mode("classification") %>%
  set_engine("xgboost")%>%
  fit(Vaccine_Hesitant~., training_set)

tuned_boost

#Variable Importance Plot 
vip::vi(tuned_boost, scale=T)

Vipplot <- tuned_boost%>%
  vip::vip()

Vipplot+
  theme_gray()

VIP_boosted <- vip::vip(tuned_boost, 
         include_type = T,  scale = T,
         aesthetics = list(color = "black", fill = "blue", size = 0.4),
         num_features = 30
        ) + 
  labs (title = "Variable Importance Plot for Boosted Trees \n The Most Important 30 Variable"
  )

VIP_boosted +
  theme(axis.text = element_text(colour = "brown", size = rel(1.3)
  ),
  title = element_text(colour = "Black", size = rel(1.5)
  ))+
  theme(plot.title = element_text(hjust = 0.5))

#Prediction

prediction_boost1 <- predict(tuned_boost, new_data = test_set, type = "class")
head (prediction_boost1)

#Confusion Matrix - Performance Metrics

##Combine predictions and truth values
pred_combined_boost <- prediction_boost1 %>%
  mutate(true_class = test_set$Vaccine_Hesitant)
head (pred_combined_boost)

##Confusion Matrix
conf_mat(data = pred_combined_boost,
         estimate = .pred_class,
         truth = true_class)

##Accuracy
accuracy(data = pred_combined_boost,
         estimate = .pred_class,
         truth = true_class)

##Specificity
spec(data = pred_combined_boost,
     estimate = .pred_class,
     truth = true_class)

###Sensitivity
sens(data = pred_combined_boost,
     estimate = .pred_class,
     truth = true_class)

##ROC AUC for RF
prediction_boost2 <- predict(tuned_boost, test_set, type="prob")%>%
  bind_cols(test_set)
prediction_boost2

roc_curve(prediction_boost2, estimate=.pred_0, truth = Vaccine_Hesitant)%>%autoplot (roc_curve)
roc_auc(prediction_boost2, estimate=.pred_0, truth = Vaccine_Hesitant)

#==========================================================================
#Model with 4 variables
tuned_boost_4<- boost_tree(
  trees = 800,
  learn_rate = .1,
  tree_depth = 1,
  sample_size = 1) %>%
  set_mode("classification") %>%
  set_engine("xgboost")%>%
  fit(Vaccine_Hesitant~ Vaccine_Trust_Index + 
        Age + Perceived_Network_Risk + Perceived_Risk, 
      training_set)

#Prediction

prediction_boost_4 <- predict(tuned_boost_4, new_data = test_set, type = "class")
head (prediction_boost_4)

#Confusion Matrix - Accuracy

##Combine predictions and truth values
pred_combined_boost_4 <- prediction_boost_4 %>%
  mutate(true_class = test_set$Vaccine_Hesitant)
head (pred_combined_boost_4)

##Confusion Matrix
conf_mat(data = pred_combined_boost_4,
         estimate = .pred_class,
         truth = true_class)

##Accuracy
accuracy(data = pred_combined_boost_4,
         estimate = .pred_class,
         truth = true_class)

##Specificity
spec(data = pred_combined_boost_4,
     estimate = .pred_class,
     truth = true_class)

##Sensitivity
sens(data = pred_combined_boost_4,
     estimate = .pred_class,
     truth = true_class)

##ROC AUC for RF
prediction_boost_4_1 <- predict(tuned_boost_4, test_set, type="prob")%>%
  bind_cols(test_set)
prediction_boost_4_1


roc_curve(prediction_boost_4_1, estimate=.pred_0, truth = Vaccine_Hesitant)%>%autoplot (roc_curve)
roc_auc(prediction_boost_4_1, estimate=.pred_0, truth = Vaccine_Hesitant)
#===============================================================================
#Model with single variable
tuned_boost_1<- boost_tree(
  trees = 800,
  learn_rate = 0.1,
  tree_depth = 1,
  sample_size = 1) %>%
  set_mode("classification") %>%
  set_engine("xgboost")%>%
  fit(Vaccine_Hesitant~ Vaccine_Trust_Index, training_set)

#Prediction

prediction_boost_1 <- predict(tuned_boost_1, new_data = test_set, type = "class")
head (prediction_boost_1)

#Confusion Matrix - Accuracy

##Combine predictions and truth values
pred_combined_boost_1 <- prediction_boost_1 %>%
  mutate(true_class = test_set$Vaccine_Hesitant)
head (pred_combined_boost_1)

##Confusion Matrix
conf_mat(data = pred_combined_boost_1,
         estimate = .pred_class,
         truth = true_class)

##Accuracy
accuracy(data = pred_combined_boost_1,
         estimate = .pred_class,
         truth = true_class)

##True Negative
spec(data = pred_combined_boost_1,
     estimate = .pred_class,
     truth = true_class)

##True Positive
sens(data = pred_combined_boost_1,
     estimate = .pred_class,
     truth = true_class)

##ROC AUC for RF
prediction_boost_1_1 <- predict(tuned_boost_1, test_set, type="prob")%>%
  bind_cols(test_set)
head (prediction_boost_1_1, 3)

roc_curve(prediction_boost_1_1, estimate=.pred_0, truth = Vaccine_Hesitant)%>%autoplot (roc_curve)
roc_auc(prediction_boost_1_1, estimate=.pred_0, truth = Vaccine_Hesitant)

##==============================================================================
#Partial Dependence Plots

set.seed(749) 
vaccine_xgb_cv <- xgboost::xgb.cv(
  data = data.matrix(subset(dataset, select = -Vaccine_Hesitant)),
  label = dataset$Vaccine_Hesitant, objective = "reg:linear", verbose = 0,
  nrounds = 1000, max_depth = 5, eta = 0.1, gamma = 0, nfold = 5,
  early_stopping_rounds = 30
)
print(vaccine_xgb_cv$best_iteration) # optimal number of trees


# Fit an XGBoost model 
set.seed(804) 
vaccine_xgb <- xgboost::xgboost(
  data = data.matrix(subset(dataset, select = -Vaccine_Hesitant)),
  label = dataset$Vaccine_Hesitant, objective = "reg:linear", verbose = 0,
  nrounds = vaccine_xgb_cv$best_iteration, max_depth = 5, eta = 0.1, gamma = 0
)

x <- data.matrix(subset(dataset, select = -Vaccine_Hesitant))  # training features
library(pdp)
p1 <- pdp:: partial(vaccine_xgb, 
                    pred.var = "Vaccine_Trust_Index", 
                    #ice = TRUE, center = TRUE,
                    #type = "auto", which.class = 1, chull= T, prob = T,
                    plot = TRUE, rug = TRUE,
                    alpha = 0.4, plot.engine = "ggplot2", 
                    train = x #, trim.outliers=T
)

p2 <- pdp:: partial(vaccine_xgb, 
                    pred.var = "Age", 
                    #ice = TRUE, center = TRUE, 
                    #type = "auto", which.class = 2, chull= T, prob = T,
                    plot = T, rug = T,  progress = "text",  
                    alpha = 0.4, plot.engine = "ggplot2", 
                    train = x #, trim.outliers=T
)

p3 <- pdp:: partial(vaccine_xgb, 
                    pred.var = "Perceived_Network_Risk", 
                    #ice = TRUE, center = TRUE, 
                    #type = "auto", which.class = 1, chull= T, prob = T,
                    plot = T, rug = T,  progress = "text",  
                    alpha = 0.4, plot.engine = "ggplot2", 
                    train = x #, trim.outliers=T
)

p4 <- pdp:: partial(vaccine_xgb, 
                    pred.var = "Perceived_Risk", 
                    #ice = TRUE, center = TRUE, 
                    #type = "auto", which.class = 1, chull= T, prob = T,
                    plot = T, rug = T,  progress = "text",  
                    alpha = 0.4, plot.engine = "ggplot2", 
                    train = x, trim.outliers=T
)

p5 <- pdp:: partial(vaccine_xgb, 
                    pred.var = "County_Cases", 
                    #ice = TRUE, center = TRUE, 
                    #type = "auto", which.class = 1, chull= T, prob = T,
                    plot = T, rug = T,  progress = "text",  
                    alpha = 0.4, plot.engine = "ggplot2", 
                    train = x #, trim.outliers=T
)
# PDPs Combined 
grid.arrange(p1, p2, p3,p4,  nrow = 2)

##========================================================================================
#Model without Vaccine Trust Index as a predictor

tuned_boost_w<- boost_tree(
  trees = 800,
  learn_rate = 0.1,
  tree_depth = 1,
  sample_size = 1) %>%
  set_mode("classification") %>%
  set_engine("xgboost")%>%
  fit(Vaccine_Hesitant~. - Vaccine_Trust_Index, training_set)

tuned_boost_w

#Prediction

prediction_boost_w <- predict(tuned_boost_w, new_data = test_set, type = "class")
head (prediction_boost_w)

#Confusion Matrix - Accuracy

##Combine predictions and truth values
pred_combined_boost_w <- prediction_boost_w %>%
  mutate(true_class = test_set$Vaccine_Hesitant)
head (pred_combined_boost_w)

##Confusion Matrix
conf_mat(data = pred_combined_boost_w,
         estimate = .pred_class,
         truth = true_class)

##Accuracy
accuracy(data = pred_combined_boost_w,
         estimate = .pred_class,
         truth = true_class)

##True Negative
spec(data = pred_combined_boost_w,
     estimate = .pred_class,
     truth = true_class)

##True Positive
sens(data = pred_combined_boost_w,
     estimate = .pred_class,
     truth = true_class)

##ROC AUC for RF
prediction_boost_w1 <- predict(tuned_boost_w, test_set, type="prob")%>%
  bind_cols(test_set)
head (prediction_boost_w1, 3)

roc_curve(prediction_boost_w1, estimate=.pred_0, truth = Vaccine_Hesitant)%>%autoplot (roc_curve)
roc_auc(prediction_boost_w1, estimate=.pred_0, truth = Vaccine_Hesitant)
