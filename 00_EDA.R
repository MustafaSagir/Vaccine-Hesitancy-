# EDA

library(readr)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(rsample)
library(skimr)
library (janitor)
library(visdat)
library("ggcorrplot")  


# Importing the dataset

vh_data15 <- read_csv("vh_data15.csv")

vh_data15%>%
  view()
# Explore the data
skim(vh_data15)
summary (vh_data15)

##Encoding Variables
vh_data15$Race = factor(vh_data15$Race, levels = c("0", "1", "2", "3"), ordered = F)
table(vh_data15$Race)                    

vh_data15$Biden = factor(vh_data15$Biden, levels = c("No","Yes"))

vh_data15$Trump = factor(vh_data15$Trump, levels = c("No", "Yes"))

vh_data15$Party_ID = factor(vh_data15$Party_ID, levels = c("Republican", "Democrat", "Independent", "Libertarian", "Other party"))

vh_data15 %>%
  count(Party_ID, Race)


## Encoding the target feature as factor
vh_data15$Vaccine_Hesitant = factor(vh_data15$Vaccine_Hesitant, levels = c(0, 1))
table(factor(vh_data15$Vaccine_Hesitant))


vh_data15 %>%
  tabyl(Vaccine_Hesitant)

vh_data15 %>%
  tabyl(Vaccine_Hesitant,Party_ID)

vh_data15 %>%
  tabyl(Vaccine_Hesitant, Race)


vh_data15 %>% 
  skim() %>% 
  dplyr::filter(n_missing > 10)

##Visualizing Missing Data

vis_miss(vh_data15, cluster = F)

vis_miss(vh_data15, cluster = T)

------------------------------------------------------------
##Removing the NAs
dataset <- vh_data15[is.na(vh_data15$Vaccine_Hesitant)==F,]
dataset <- na.omit(dataset)

dataset <- na.omit (vh_data15)

glimpse(dataset)
skim_tee(dataset)


#Data Visualization

##Barplots

dataset %>%
  ggplot(aes(x=Vaccine_Hesitant))+
  geom_bar(width=0.7, fill="steelblue")+
  coord_flip()+
  theme_minimal()


dataset %>%
  ggplot(aes(x = Vaccine_Hesitant, fill = Party_ID)) +
  geom_bar(width = 0.25)
 

dataset %>%
  ggplot(aes(x = Vaccine_Hesitant, fill = Race)) +
  geom_bar(width = 0.35) 


##Boxplots

dataset %>%
  ggplot(aes(x = Vaccine_Hesitant, y = Vaccine_Trust_Index)) +
  geom_boxplot(varwidth=T, fill="red") 

dataset %>%
  ggplot(aes( x= Race, y = Vaccine_Trust_Index)) +
  geom_boxplot(varwidth=T, fill="plum") 


dataset %>%
  ggplot(aes( x= Party_ID, y = Vaccine_Trust_Index)) +
  geom_boxplot(varwidth=T, fill="plum") 


dataset %>%
  ggplot(aes(x = Vaccine_Hesitant, y = Trust_Media)) +
  geom_boxplot(varwidth=T, fill="orange") 


dataset %>%
  ggplot(aes(x = Vaccine_Hesitant, y = Trust_Science_Apolitical)) +
  geom_boxplot (varwidth=T, fill="steelblue") 

## Correlation Matrix        

cols_for_corr <- dataset%>% 
  select(County_Density, Vaccine_Trust_Index, Personal_Responsibility, Trust_Science_Apolitical, 
         Vaccine_Trust_Index, Trust_Science_Politicians,Trust_Science_Media, Trust_Science_Community,
         Trust_Science_Politicians,Trust_Science_Media, Trust_Science_Community,
         Trust_National, Trust_State, Trust_Local, Trust_Media, Perceived_Risk)
                  

corr <- round(cor(cols_for_corr),1)
corr

p.mat <- cor_pmat(cols_for_corr)
p.mat

ggcorrplot(corr, hc.order = TRUE, type = "lower", lab = TRUE,
           outline.color = "white", ggtheme = ggplot2::theme_gray,
           colors = c("#6D9EC1", "white", "#E46726"))


------------------------------------------------------------
#Data Partition
library(tidymodels)
set.seed(1234)
dataset_split <- initial_split (vh_data15, prop=0.75, strata = Vaccine_Hesitant)
training_set<- training(dataset_split)
test_set <- testing (dataset_split)

#Imputing the NAs
dataset_recipe <- recipe(Vaccine_Hesitant ~., data = training_set)%>%
  step_impute_bag(all_predictors())


caret:: nearZeroVar(training_set, saveMetrics = T)%>%
  rownames_to_column()%>%
  filter(nzv)

dataset_recipe %>%
  step_nzv(all_predictors())%>%
  step_pca(all_numeric(), threshold = .95, -all_outcomes())

prepare <- prep(dataset_recipe, training = training_set)
prepare

baked_train <- bake (prepare, new_data = training_set)
baked_test <- bake (prepare, new_data = test_set)

baked_train
baked_test




