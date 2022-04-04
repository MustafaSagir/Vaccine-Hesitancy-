#Bagged Trees

# Importing the dataset

library(readr)
vh_data15 <- read_csv("vh_data15.csv")

summary (vh_data15)

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

glimpse(dataset)
skim(dataset)
#Data Partition
library(tidymodels)
set.seed(1234)
dataset_split <- initial_split (dataset, prop=0.75, strata = Vaccine_Hesitant)
training_set<- training(dataset_split)
test_set <- testing (dataset_split)

#Bagged Trees
#install.packages("baguette")
library(baguette)
spec_bagged <- baguette::bag_tree() %>%
  set_mode("classification") %>%
  set_engine("rpart", times = 100)

model_bagged <- fit(spec_bagged, 
                    formula = Vaccine_Hesitant ~ ., 
                    data = training_set)
model_bagged

#Cross Validation

set.seed(2345)
my_folds <- vfold_cv(training_set, v = 10)

fits_cv <- fit_resamples(spec_bagged,
                         Vaccine_Hesitant ~ .,
                         resamples = my_folds,
                         metrics = metric_set(accuracy, roc_auc))

fits_cv

## Collect accuracy and ROC_AUC of all model runs
all_roc_auc <- collect_metrics(fits_cv,
                               summarize = FALSE)
print(all_roc_auc)

library(ggplot2)
ggplot(all_roc_auc, aes(x = .estimate,
                        fill = .metric)) +
  geom_histogram()

#Collect and summarize accuracy of all model runs
collect_metrics(fits_cv)

#Tuning

doParallel::registerDoParallel()

spec_untuned_bagged <- baguette::bag_tree(tree_depth =tune(), cost_complexity=tune()) %>%
  set_mode("classification") %>%
  set_engine("rpart", times = 100)

spec_untuned_bagged

bagged_grid <- grid_regular(
  parameters(spec_untuned_bagged),
  levels = 3)

tune_results <- tune_grid(
  spec_untuned_bagged,
  Vaccine_Hesitant ~ .,
  resamples = vfold_cv(training_set, v = 4),
  grid = bagged_grid,
  metrics = metric_set(accuracy, roc_auc))

tune_results
##Visualizing tuning results
autoplot(tune_results)

# Select the best performing parameters
final_params <- select_best(tune_results, metric = "roc_auc")
final_params1 <- select_best(tune_results, metric = "accuracy")

final_params
final_params1

# Plug them into the specification
best_spec <- finalize_model(spec_untuned_bagged,
                            final_params)
best_spec

#Tuned Bagged Tree
spec_tuned_bag <- baguette::bag_tree(min_n=2, tree_depth =8, cost_complexity=3.16227766016838e-06) %>%
  set_mode("classification") %>%
  set_engine("rpart", times = 100)
  
  
tuned_bag <- fit(spec_tuned_bag, formula = Vaccine_Hesitant ~ ., data = training_set)

tuned_bag

##rediction

prediction_bag <- predict(tuned_bag, new_data = test_set, type = "class")
head (prediction_bag,3)

##Confusion Matrix - Accuracy

###Combine predictions and truth values
pred_combined_bag <- prediction_bag %>%
  mutate(true_class = test_set$Vaccine_Hesitant)
pred_combined_bag


###Confusion Matrix
conf_mat(data = pred_combined_bag,
         estimate = .pred_class,
         truth = true_class)

###Accuracy
accuracy(data = pred_combined_bag,
         estimate = .pred_class,
         truth = true_class)

###Specificity
spec(data = pred_combined_bag,
     estimate = .pred_class,
     truth = true_class)

###Sensitivity
sens(data = pred_combined_bag,
     estimate = .pred_class,
     truth = true_class)


## Calculate the ROC curve for all thresholds
prediction2 <- predict(tuned_bag, new_data = test_set, type = "prob")%>%
  bind_cols(test_set)
prediction2

##ROC
roc <- roc_curve(prediction2,
                 estimate = .pred_0,
                 truth = Vaccine_Hesitant)
## Plot the ROC curve
autoplot(roc)

##Calculate area under curve
roc_auc(prediction2,
        estimate = .pred_0,
        truth = Vaccine_Hesitant)


# Variable Importance Plot
library(caret)
bagged_2 <- train (
  Vaccine_Hesitant~.,
  data = training_set,
  method = "treebag",
  trControl = trainControl (method = "cv", number = 10),
  nbagg = 200,
  control = rpart.control(minsplit = 2, cp=0)
)

vi(bagged_2, scale = T)

VIP_bagged <-vip::vip(
  bagged_2, 
  include_type = T, scale = T,
  aesthetics = list(color = "black", fill = "blue", size = 0.8)) +
  labs (title = "Variable Importance Plot for Bagged Trees \n The Most Important 10 Variable"
  )

VIP_bagged +
  theme(axis.text = element_text(colour = "brown", size = rel(1.3) #, angle = 45
  ),
  title = element_text(colour = "Black", size = rel(1.5)
  ))+
  theme(plot.title = element_text(hjust = 0.5))


#Partial Dependence Plots
library(pdp)
pdp_VTI <- partial (bagged_2, "Vaccine_Trust_Index", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

pdp_TSC <- partial (bagged_2, "Trust_Science_Community", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

pdp_DC <- partial (bagged_2, "Doctor_Comfort", grid.resolution = 20, prob = T,
                   chull= T, plot = T, plot.engine= "ggplot2")

pdp_PNR <- partial (bagged_2, "Perceived_Network_Risk", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

pdp_AGE <- partial (bagged_2, "Age", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

gridExtra::grid.arrange(pdp_VTI, pdp_TSC, pdp_DC, pdp_PNR,pdp_AGE,   nrow=2)


















