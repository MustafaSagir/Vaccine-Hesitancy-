#Logistic Regression without weights

library(tidyverse)
library(dplyr)
library(broom)
library (janitor)
library(ggplot2)
library(rsample)
library(caret)
library(ROCR)
library(vip)
library(pdp)

# Importing the dataset
library(readr)
vh_data14 <- read_csv("vh_data14.csv")

# Explore the data
summary (vh_data14)
skimr::skim(vh_data14)
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

## Removing unnecessary variables and NAs
vh_data14 <- vh_data14 %>%
  select(-round, - respondent_id, -response_round_2, -obs, -inv_p)

vh_data14 <- na.omit(vh_data14)

vh_data14 %>%
  tabyl(vaccine_hesitant)

##Logistic regression without weights
set.seed(1234)
vaccine_without_weights <- glm (
  formula = vaccine_hesitant ~ ., 
  family = "binomial", 
  na.action = na.omit,
  data = vh_data14)

summary(vaccine_without_weights)

library(broom)
#Create dataframe with results
results<- as.data.frame(tidy(vaccine_without_weights))
#results2 <- as.data.frame(glance(vaccine_without_weights))

#Export as csv
write.csv(results, "results.csv")
#write.csv(results2, "results2.csv")

#Data Partition
library(tidymodels)
set.seed(2345)
dataset_split <- initial_split (vh_data14, prop=0.75, strata = vaccine_hesitant)
training_set<- training(dataset_split)
test_set <- testing (dataset_split)

#Logistic regression Prediction without weights on Training Set
set.seed(4567)
Vaccine_lr_without_weights <- glm (
  formula= vaccine_hesitant ~ ., 
  family = "binomial",
  na.action = na.exclude,
  data = training_set
  )
summary (Vaccine_lr_without_weights)

#Assessing model accuracy
set.seed (3456)
cv_Vaccine_lr_without_weights <- train(
  vaccine_hesitant ~ .,  
  data = training_set,
  method= "glm",
  family= stats::binomial(link = "logit"),
  na.action = na.exclude,
  trControl= trainControl(method = "cv", number = 10)
)

summary (cv_Vaccine_lr_without_weights)

#Predict class
pred_class_unweighted <- predict (cv_Vaccine_lr_without_weights, training_set)
head (pred_class_unweighted)

table (pred_class_unweighted)

#Confusion Matrix
confusionMatrix(
  data = relevel(pred_class_unweighted, ref = "1"),
  reference = relevel(training_set$vaccine_hesitant, ref = "1")
)

#ROC Curve
library(ROCR)

Vaccine_prob_unweighted <- predict (cv_Vaccine_lr_without_weights, type = "prob")$"1"
head (Vaccine_prob_unweighted)

#Compute AUC metrics and Plot ROC curve
perf_Vac <- prediction(Vaccine_prob_unweighted, training_set$vaccine_hesitant)%>%
  performance(measure = "tpr", x.measure = "fpr")

plot (perf_Vac, col = "blue")


# 10 Fold CV on a PLS model tuning the number of PCs 
set.seed(5678)
cv_Vaccine_pls_unweighted <- train(
  vaccine_hesitant ~ ., 
  data = training_set,
  method= "pls",
  family= "binomial",
  preProcess = c("zv", "center", "scale"),
  trControl= trainControl(method = "cv", number = 10),
  tuneLength = 36
)

cv_Vaccine_pls_unweighted

cv_Vaccine_pls_unweighted$bestTune
ggplot(cv_Vaccine_pls_unweighted)

#Variable Importance Plot
vip(cv_Vaccine_pls_unweighted, num_features = 10)

#Partial Dependence Plots
library(pdp)

pdp_VTI <- partial (cv_Vaccine_pls_unweighted, "vaccine_trust", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

pdp_TSC <- partial (cv_Vaccine_pls_unweighted, "trust_science_community", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

pdp_DC <- partial (cv_Vaccine_pls_unweighted, "doctor_comfort", grid.resolution = 20, prob = T,
                   chull= T, plot = T, plot.engine= "ggplot2")

pdp_PNR <- partial (cv_Vaccine_pls_unweighted, "perceived_network_risk", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

pdp_AGE <- partial (cv_Vaccine_pls_unweighted, "age", grid.resolution = 20, prob = T,
                    chull= T, plot = T, plot.engine= "ggplot2")

gridExtra::grid.arrange(pdp_VTI, pdp_TSC, pdp_DC, pdp_PNR,pdp_AGE,   nrow=2)

# Tuned Model
set.seed(6789)
Vaccine_tuned_unweighted <- train(
  vaccine_hesitant~ vaccine_trust + trust_science_community + doctor_comfort +
    perceived_network_risk +age, 
  data = training_set,
  method= "glm",
  family= "binomial",
  preProcess = c("zv", "center", "scale")
)

#Predictions on Test Set

#Predict class
pred_class_unweighted <- predict (Vaccine_tuned_unweighted, test_set)
head (pred_class_unweighted)

#Confusion Matrix
confusionMatrix(
  data = relevel(pred_class_unweighted, ref = "1"),
  reference = relevel(test_set$vaccine_hesitant, ref = "1")
)

#ROC Curve
#ROC AUC for RF
prediction_2 <- predict(Vaccine_tuned_unweighted, test_set, type="prob")%>%
  bind_cols(test_set)
head(prediction_2)

roc_curve(prediction_2, estimate='0', truth = vaccine_hesitant)%>%
  autoplot (roc_curve)

roc_auc(prediction_2, estimate='0', truth = vaccine_hesitant)

#=======================================================================
# Tuned Model Using all variables 
set.seed(7891)
Vaccine_tuned_all <- train(
  vaccine_hesitant ~ vaccine_trust +
    county_density + individual_responsibility + trust_gov_nat + trust_gov_state + 
    trust_gov_local + trust_media + perceived_personal_risk + 
    perceived_network_risk + doctor_comfort + fear_needles + 
    condition_pregnant + condition_asthma + 
    condition_lung + condition_diabetes + condition_immune + condition_obesity + 
    condition_heart + condition_organ + 
    male + race + age + psindex + nsindex + college + pandemic_impact_personal + 
    pandemic_impact_network + infected_personal + biden  + trump +
    infected_network + trust_science_polmotives + trust_science_politicians + 
    trust_science_media + trust_science_community + party_id + income + evangelical, 
  data = training_set,
  method= "glm",
  family= "binomial",
  preProcess = c("zv", "center", "scale")
)

#Predictions on Test Set

#Predict class
pred_class_tuned_all <- predict (Vaccine_tuned_all, test_set)
head (pred_class_tuned_all)

#Confusion Matrix
confusionMatrix(
  data = relevel(pred_class_tuned_all, ref = "0"),
  reference = relevel(test_set$vaccine_hesitant, ref = "0")
)

#ROC-AUC 
prediction_2_full <- predict(Vaccine_tuned_all, test_set, type="prob")%>%
  bind_cols(test_set)
head(prediction_2_full)

roc_curve(prediction_2_full, estimate='0', truth = vaccine_hesitant)%>%
  autoplot (roc_curve)

roc_auc(prediction_2_full, estimate='0', truth = vaccine_hesitant)

# Variable Importance Plot All Model
VIP_LR <-vip::vip(
  Vaccine_tuned_all, 
  include_type = T, scale = T,
  aesthetics = list(color = "black", fill = "blue", size = 0.8)) +
  labs (title = "Variable Importance Plot for Logistic regression \n The Most Important 10 Variable"
  )


VIP_LR +
  theme(axis.text = element_text(colour = "brown", size = rel(1.3)),
        title = element_text(colour = "Black", size = rel(1.5)))+
  theme(plot.title = element_text(hjust = 0.5))

