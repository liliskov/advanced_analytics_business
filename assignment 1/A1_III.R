# Advanced Analytics in a Big Data World -----------------------------------------------------------------------------------
# Description: Assignment 1. Part III.
#   Business problem: Who are my 20 predicted churners with highest retained profit?
#         => proxy for profitability: 'average cost min'
# Author: Lili Vandermeersch r0691855
#
# File History---------------------------
# Creation: 02/O4/24
#
#
# Paths ----------------------------------------------------------------------------------------------------
rm(list =ls()) # clears R environment
options(scipen=999) # disables scientific notation

# Packages -------------------------------------------------------------------------------------------------
library(here)
library(rio)        
library(dplyr)      
library(reshape2)
library(caret)
library(rpart)
library(pROC)
library(ggplot2) 
library(boot)
library(glmnet)

# Importing the data -----------------------------------------------------------------------------------------
train_df <- import(here("train.csv"))
test_df <- import(here("test.csv"))

##############################################################################################
### Data Exploration and Manipulation  ###
##############################################################################################
# Data Exploration -------------------------------------------------------------------------------------------
nrow(train_df)      # number of obs
ncol(train_df)      # number of variables
names(train_df)     # names of variables

head(train_df)
summary(train_df)
str(train_df)

# Data Types -------------------------------------------------------------------------------------------
categorical_vars <- c("Gender", "tariff", "Handset", "Usage_Band", "Tariff_OK", "high Dropped calls", "No Usage")
train_df[categorical_vars] <- lapply(train_df[categorical_vars], as.factor)

train_df$target <- as.factor(train_df$target)

# Removing the 'id' column from the dataset
train_df <- select(train_df, -id)

# Feature Engineering ------------------------------------------------------------------------------------------
# Converting Connect_Date to Date type
train_df$Connect_Date <- as.Date(train_df$Connect_Date, format = "%d/%m/%y")

# Feature engineering: Extracting year, month, and tenure in days 
reference_date <- max(train_df$Connect_Date, na.rm = TRUE) + 1
train_df$Tenure_days <- as.numeric(difftime(reference_date, train_df$Connect_Date, units = "days"))
train_df$Connect_Year <- as.numeric(format(train_df$Connect_Date, "%Y"))
train_df$Connect_Month <- as.numeric(format(train_df$Connect_Date, "%m"))

# Checking for missing values  -------------------------------------------------------------------------------------------
sum(is.na(train_df))
colSums(is.na(train_df))

# Handling missing values -------------------------------------------------------------------------------------
# For numerical columns, filling missing values with the median
num_cols <- sapply(train_df, is.numeric)
train_df[, num_cols] <- lapply(train_df[, num_cols], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# For factors, filling missing values with the mode
getMode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
modeValue <- getMode(train_df$Usage_Band)
train_df$Usage_Band[is.na(train_df$Usage_Band)] <- modeValue

train_df <- train_df %>%
  select(-Connect_Date)

# Splitting the training data into quartiles for different models --------------------------------------------------------------------------
quartiles <- quantile(train_df$`average cost min`, probs = c(0.25, 0.5, 0.75))
train_low <- train_df[train_df$`average cost min` <= quartiles[1], ]
train_medium <- train_df[train_df$`average cost min` > quartiles[1] & train_df$`average cost min` <= quartiles[2], ]
train_high <- train_df[train_df$`average cost min` > quartiles[3], ]

##############################################################################################
### PCA  ###
##############################################################################################
# Select numeric variables, excluding the target variable
numericVars <- train_df %>% select(where(is.numeric))

# Standardize the data
preProcValues <- preProcess(numericVars, method = c("center", "scale"))
numericVarsStd <- predict(preProcValues, numericVars)

# Perform PCA
pcaResults <- prcomp(numericVarsStd)

# Define function to process each group
prepare_group_data <- function(data, preProcValues, pcaThreshold = 1) {
  # Standardize numeric variables
  numericVars <- select(data, where(is.numeric))
  numericVarsStd <- predict(preProcValues, numericVars)
  
  # Apply PCA
  pcaResults <- prcomp(numericVarsStd, center = TRUE, scale. = TRUE)
  
  # Decide on the number of components to keep
  componentsToRetain <- sum(pcaResults$sdev^2 > pcaThreshold)
  
  # Extract the selected components
  pcaData <- as.data.frame(pcaResults$x[, 1:componentsToRetain])
  
  # Combine PCA components with categorical variables
  factorVars <- select(data, where(is.factor))
  combinedData <- cbind(pcaData, factorVars)
  
  return(list(combinedData = combinedData, pcaResults = pcaResults))
}

# Apply the function to low, medium, and high groups
data_low <- prepare_group_data(train_low, preProcValues)
data_medium <- prepare_group_data(train_medium, preProcValues)
data_high <- prepare_group_data(train_high, preProcValues)

# Now you have combinedData and pcaResults for each group
combinedData_low <- data_low$combinedData
pcaResults_low <- data_low$pcaResults

combinedData_medium <- data_medium$combinedData
pcaResults_medium <- data_medium$pcaResults

combinedData_high <- data_high$combinedData
pcaResults_high <- data_high$pcaResults

# Decide on the number of components to keep (demonstration purpose, adjust as needed)
componentsToRetain_low <- sum(pcaResults_low$sdev^2 > 1)
componentsToRetain_medium <- sum(pcaResults_medium$sdev^2 > 1)
componentsToRetain_high <- sum(pcaResults_high$sdev^2 > 1)

# Combine PCA components with categorical variables
pcaData_low <- as.data.frame(pcaResults_low$x[, 1:componentsToRetain_low])
factorVars_low <- select(train_low, where(is.factor))
combinedData_low <- cbind(pcaData_low, factorVars_low)

pcaData_medium <- as.data.frame(pcaResults_medium$x[, 1:componentsToRetain_medium])
factorVars_medium <- select(train_medium, where(is.factor))
combinedData_medium <- cbind(pcaData_medium, factorVars_medium)

pcaData_high <- as.data.frame(pcaResults_high$x[, 1:componentsToRetain_high])
factorVars_high <- select(train_high, where(is.factor))
combinedData_high <- cbind(pcaData_high, factorVars_high)

##############################################################################################
### Logistic Regression with Combined Data  ###
##############################################################################################
# Fit logistic regression models for each group
logitModel_low <- glm(target ~ ., data = combinedData_low, family = binomial(link = "logit"))
logitModel_medium <- glm(target ~ ., data = combinedData_medium, family = binomial(link = "logit"))
logitModel_high <- glm(target ~ ., data = combinedData_high, family = binomial(link = "logit"))

# Summary of the logistic regression models
summary(logitModel_low)
summary(logitModel_medium)
summary(logitModel_high)

# Prepare validation data for each group
validationData_low <- prepare_group_data(validation_low, preProcValues)
validationData_medium <- prepare_group_data(validation_medium, preProcValues)
validationData_high <- prepare_group_data(validation_high, preProcValues)

# Combine PCA with categorical variables for each validation set
validation_combinedData_low <- validationData_low$combinedData
validation_combinedData_medium <- validationData_medium$combinedData
validation_combinedData_high <- validationData_high$combinedData

# Make predictions for each group
validation_probabilities_low <- predict(logitModel_low, newdata = validation_combinedData_low, type = "response")
validation_probabilities_medium <- predict(logitModel_medium, newdata = validation_combinedData_medium, type = "response")
validation_probabilities_high <- predict(logitModel_high, newdata = validation_combinedData_high, type = "response")

# Evaluate model performance for each group (example for low group, repeat for others)
roc_curve_low <- roc(response = validation_low$target, predictor = validation_probabilities_low)
auc_value_low <- auc(roc_curve_low)
print(paste("AUC for low-value group:", auc_value_low))

# Convert predictions and actual values to factors if they aren't already for sensitivity calculation etc.
# Repeat AUC calculation and other performance metrics evaluation for medium and high groups