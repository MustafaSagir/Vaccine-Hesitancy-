#Logistic Regression with weights

library(tidyverse)
library(dplyr)
library(ggplot2)
library(rsample)
library(caret)
library(ROCR)
library(vip)
library(pdp)
library(broom)
library(skimr)
library (janitor)
library(ROCR)

# Importing the dataset
library(readr)
vh_data14 <- read_csv("vh_data14.csv")

vh_data14%>%
  view()
# Explore the data
summary (vh_data14)
skimr::skim(vh_data14)
##Encoding Variables
vh_data14$race = factor(vh_data14$race, levels = c("1", "2", "3", "4"), ordered = F)
table(vh_data14$race)                    

vh_data14<-vh_data14 %>%
  rename(biden = president_approval, 
         trump = trump_approval_retrospective,
         perceived_personal_risk = perceived_personal_riskq297_4, 
         wghts = inv_p)


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
  select(-round, - respondent_id, -response_round_2)

vh_data14 <- na.omit(vh_data14)

library (janitor)
vh_data14 %>%
  tabyl(vaccine_hesitant, race)

##Logistic regression with weights

Predictors <- c("vaccine_trust", "county_density", "individual_responsibility", 
                "trust_gov_nat", "trust_gov_state", 
                "trust_gov_local", "trust_media", "perceived_personal_risk", 
                "perceived_network_risk", "doctor_comfort", "fear_needles", 
                "condition_pregnant", "condition_asthma", 
                "condition_lung", "condition_diabetes", "condition_immune", "condition_obesity", 
                "condition_heart", "condition_organ", "county_covid_cap_cases", 
                "county_covid_cap_cases2wk", "male", "race", "age", "psindex", "nsindex", 
                "college", "pandemic_impact_personal", "pandemic_impact_network", 
                "infected_personal", "biden", "trump", "infected_network", 
                "trust_science_polmotives", 
                "trust_science_politicians", "trust_science_media", 
                "trust_science_community", "party_id", "income, evangelical")

set.seed(1234)
vaccine_with_weights <- glm (
  formula = vaccine_hesitant ~ .-wghts, 
  #family = "binomial", 
  family = stats::quasibinomial(link = "logit"),
  na.action = na.exclude,
  weights = wghts,
  data = vh_data14)

summary(vaccine_with_weights)

#Create dataframe with results
library(broom)

results2<- as.data.frame(tidy(vaccine_with_weights))

#Export as csv
write.csv(results2, "results2.csv")

#Data Partition
library(tidymodels)
set.seed(2345)
dataset_split <- initial_split (vh_data14, prop=0.75, strata = vaccine_hesitant)
training_set<- training(dataset_split)
test_set <- testing (dataset_split)

#Logistic regression Prediction with weights on Training Set
set.seed(3456)
Vaccine_lr_with_weights <- glm (
  formula= vaccine_hesitant ~ .-wghts, 
  family = stats::quasibinomial(link = "logit"),
  na.action = na.exclude,
  weights = wghts,
  data = training_set
)
summary (Vaccine_lr_with_weights)

#Assessing model accuracy
set.seed (3456)
cv_Vaccine_lr_with_weights <- train(
  vaccine_hesitant ~ .-wghts,  
  data = training_set,
  method= "glm",
  family = stats::quasibinomial(link = "logit"),
  na.action = na.exclude,
  weights = wghts,
  trControl= trainControl(method = "cv", number = 10)
)

summary (cv_Vaccine_lr_with_weights)

#Predict class
pred_class_weighted <- predict (cv_Vaccine_lr_with_weights, training_set)
head (pred_class_weighted)

table (pred_class_weighted)

#Confusion Matrix
confusionMatrix(
  data = relevel(pred_class_weighted, ref = "1"),
  reference = relevel(training_set$vaccine_hesitant, ref = "1")
)

#ROC Curve
library(ROCR)
Vaccine_prob_weighted <- predict (cv_Vaccine_lr_with_weights, type = "prob")$"1"
head (Vaccine_prob_weighted)

#Compute AUC metrics and Plot ROC curve
perf_Vac <- prediction(Vaccine_prob_weighted, training_set$vaccine_hesitant)%>%
  performance(measure = "tpr", x.measure = "fpr")

plot (perf_Vac, col = "red")


# 10 Fold CV on a PLS model tuning the number of PCs 
set.seed(5678)
cv_Vaccine_pls_weighted <- train(
  vaccine_hesitant ~ .-wghts, 
  data = training_set,
  method= "pls",
  family = stats::quasibinomial(link = "logit"),
  na.action = na.exclude,
  weights = wghts,
  preProcess = c("zv", "center", "scale"),
  trControl= trainControl(method = "cv", number = 10),
  tuneLength = 36
)

cv_Vaccine_pls_weighted

cv_Vaccine_pls_weighted$bestTune
ggplot(cv_Vaccine_pls_weighted)

#Variable Importance Plot
vip(cv_Vaccine_pls_weighted, num_features = 10)


#Partial Dependence Plots
library(pdp)

pdp_VTI <- pdp:: partial (cv_Vaccine_pls_weighted, "vaccine_trust", grid.resolution = 20, prob = T,
                    chull= T, which.class = 2, plot = T, plot.engine= "ggplot2")

pdp_TSC <- pdp:: partial (cv_Vaccine_pls_weighted, "trust_science_community", grid.resolution = 20, prob = T,
                    chull= T, which.class = 2, plot = T, plot.engine= "ggplot2")

pdp_DC <- pdp:: partial (cv_Vaccine_pls_weighted, "doctor_comfort", grid.resolution = 20, prob = T,
                   chull= T, which.class = 2, plot = T, plot.engine= "ggplot2")

pdp_PNR <- pdp:: partial (cv_Vaccine_pls_weighted, "perceived_network_risk", grid.resolution = 20, prob = T,
                    chull= T, which.class = 2, plot = T, plot.engine= "ggplot2", progress = "text")

pdp_AGE <- pdp:: partial (cv_Vaccine_pls_weighted, "age", grid.resolution = 20, prob = T,
                    chull= T, which.class = 2, plot = T, alpha = .4, plot.engine= "ggplot2")

gridExtra::grid.arrange(pdp_VTI, pdp_TSC, pdp_DC, pdp_PNR,pdp_AGE,   nrow=2)

# Tuned Model
set.seed(6789)
Vaccine_tuned_weighted <- train(
  vaccine_hesitant~ vaccine_trust + trust_science_community + doctor_comfort +
    perceived_network_risk + age, 
  data = training_set,
  method= "glm",
  family = stats::quasibinomial(link = "logit"),
  na.action = na.exclude,
  weights = wghts,
  preProcess = c("zv", "center", "scale")
)


vip(Vaccine_tuned_weighted)

#Predictions on Test/Validation Set

#Predict class
pred_class_weighted_tuned <- predict (Vaccine_tuned_weighted, test_set)
head (pred_class_weighted_tuned)

#Confusion Matrix
confusionMatrix(
  data = relevel(pred_class_weighted_tuned, ref = "1"),
  reference = relevel(test_set$vaccine_hesitant, ref = "1")
)

#ROC Curve
#ROC AUC for RF
prediction_2 <- predict(Vaccine_tuned_weighted, test_set, type="prob")%>%
  bind_cols(test_set)
head(prediction_2)

roc_curve(prediction_2, estimate='0', truth = vaccine_hesitant)%>%
  autoplot (roc_curve)

roc_auc(prediction_2, estimate='0', truth = vaccine_hesitant)

# Tuned Model Using all variables 
set.seed(7891)
Vaccine_tuned_all <- train(
  vaccine_hesitant~ .-wghts,  
  data = training_set,
  method= "glm",
  family = stats::quasibinomial(link = "logit"),
  na.action = na.exclude,
  weights = wghts,
  preProcess = c("zv", "center", "scale")
)

#Predictions on Test Set

#Predict class
pred_class_full <- predict (Vaccine_tuned_all, test_set)
head (pred_class_full)

#Confusion Matrix
confusionMatrix(
  data = relevel(pred_class_full, ref = "1"),
  reference = relevel(test_set$vaccine_hesitant, ref = "1")
)

#ROC Curve
#ROC AUC for RF
prediction_2_full <- predict(Vaccine_tuned_all, test_set, type="prob")%>%
  bind_cols(test_set)
head(prediction_2_full)

roc_curve(prediction_2_full, estimate='0', truth = vaccine_hesitant)%>%
  autoplot (roc_curve)

roc_auc(prediction_2_full, estimate='0', truth = vaccine_hesitant)
