# Random Forest 

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
skimr::skim(dataset)

#Data Partition
library(tidymodels)
set.seed(1234)
dataset_split <- initial_split (dataset, prop=0.75, strata = Vaccine_Hesitant)
training_set<- training(dataset_split)
test_set <- testing (dataset_split)

#Random Forest
library(tidymodels)
spec_RF<- rand_forest(mode = "classification") %>% 
  set_engine("ranger", importance = "impurity")

model_RF<-spec_RF%>%
  fit(Vaccine_Hesitant~., training_set)

model_RF

#Cross Validation
set.seed(1234)
my_folds <- vfold_cv(training_set, v = 10)

fits_cv <- fit_resamples(spec_RF,
                         Vaccine_Hesitant ~ .,
                         resamples = my_folds,
                         metrics = metric_set(accuracy, roc_auc))

fits_cv

## Collect accuracy and ROC_AUC of all model runs
all_metrics <- collect_metrics(fits_cv,
                                summarize = FALSE)
print(all_metrics)

library(ggplot2)
ggplot(all_metrics, aes(x = .estimate,
                        fill = .metric)) +
                       geom_histogram()

#Collect and summarize accuracy of all model runs
collect_metrics(fits_cv)

#Tuning
doParallel::registerDoParallel()

spec_untuned_RF <- rand_forest(trees= tune(), mode= "classification", min_n = tune()) %>%
  set_engine("ranger", importance = "impurity")
  
RF_grid <- grid_regular(
  parameters(spec_untuned_RF),
  levels = 4)

tune_results <- tune_grid(
  spec_untuned_RF,
  Vaccine_Hesitant ~ .,
  resamples = vfold_cv(training_set, v = 10),
  grid = RF_grid,
  metrics = metric_set(accuracy, roc_auc))

tune_results

##Visualizing tuning results
autoplot(tune_results)

# Select the best performing parameters
final_params <- select_best(tune_results)
final_params1 <- select_best(tune_results, metric = "roc_auc")

final_params
final_params1

# Plug them into the specification
best_spec <- finalize_model(spec_untuned_RF,
                            final_params)
best_spec

#Tuned Random Forest
tuned_RF<-rand_forest(min_n = 27, trees = 1333, mode = "classification")%>%
  set_engine("ranger", importance = "impurity")%>%
  fit(Vaccine_Hesitant~., training_set)

#Variable Importance Plot 
VIP_RF <-vip::vip(
  tuned_RF, 
  include_type = T, scale = T,
  aesthetics = list(color = "black", fill = "blue", size = 0.8)) +
  labs (title = "Variable Importance Plot for Random Forest \n The Most Important 10 Variable"
  )

VIP_RF +
  theme(axis.text = element_text(colour = "brown", size = rel(1.3)),
  title = element_text(colour = "Black", size = rel(1.5)))+
  theme(plot.title = element_text(hjust = 0.5))

#Prediction
prediction_RF1 <- predict(tuned_RF, new_data = test_set, type = "class")
head (prediction_RF1)

##Confusion Matrix - Performance Metrics

##Combine predictions and truth values
pred_combined_RF <- prediction_RF1 %>%
  mutate(true_class = test_set$Vaccine_Hesitant)
head (pred_combined_RF)

##Confusion Matrix
conf_mat(data = pred_combined_RF,
         estimate = .pred_class,
         truth = true_class)

##Accuracy
accuracy(data = pred_combined_RF,
         estimate = .pred_class,
         truth = true_class)

## Specificity (True Negative)
spec(data = pred_combined_RF,
     estimate = .pred_class,
     truth = true_class)

## Sensitivity (True Positive)
sens(data = pred_combined_RF,
     estimate = .pred_class,
     truth = true_class)

##ROC AUC for RF
prediction_RF2 <- predict(tuned_RF, test_set, type="prob")%>%
  bind_cols(test_set)
prediction_RF2

roc_curve(prediction_RF2, estimate=.pred_0, truth = Vaccine_Hesitant)%>%autoplot (roc_curve)
roc_auc(prediction_RF2, estimate=.pred_0, truth = Vaccine_Hesitant)

#PDPs

RF2 <- randomForest::randomForest(Vaccine_Hesitant ~ ., data=training_set, importance=TRUE,
                                  proximity=TRUE, ntree= 1333)

library(pdp)
pdp_VTI <- partial (RF2, "Vaccine_Trust_Index", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

pdp_TSC <- partial (RF2, "Trust_Science_Community", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

pdp_DC <- partial (RF2, "Doctor_Comfort", grid.resolution = 20, prob = T,
                   chull= T, plot = T, plot.engine= "ggplot2")

pdp_PNR <- partial (RF2, "Perceived_Network_Risk", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

pdp_AGE <- partial (RF2, "Age", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

gridExtra::grid.arrange(pdp_VTI, pdp_TSC, pdp_DC, pdp_PNR,pdp_AGE,   nrow=2)




