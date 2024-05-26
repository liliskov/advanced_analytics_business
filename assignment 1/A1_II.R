# Advanced Analytics in a Big Data World -----------------------------------------------------------------------------------
# Description: Assignment 1. Part II.
#   Business problem: Who are my 20 predicted churners with highest retained profit?
#         => proxy for profitability: avg_cost_min
# Author: Lili Vandermeersch r0691855
#
# File History---------------------------
# Creation: 01/O4/24
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
library(gridExtra)
library(tidyverse)
library(ggplot2)
library(GGally)
library(smotefamily)
library(randomForest)
library(rpart)
library(rpart.plot)
library(e1071)

# Importing the data -----------------------------------------------------------------------------------------
train_df <- import(here("train.csv"))
test_df <- import(here("test.csv"))

names(train_df) <- gsub(" ", "_", names(train_df))
names(test_df) <- gsub(" ", "_", names(test_df))

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
categorical_vars <- c("Gender", "tariff", "Handset", "Usage_Band", "Tariff_OK", "high_Dropped_calls", "No_Usage")
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

# Splitting Data into Training and Validation Sets ------------------------------------------------------------------------------------------
set.seed(200) 
indexes <- createDataPartition(y = train_df$target, p = 0.8, list = FALSE)
train_data <- train_df[indexes, ]
validation_data <- train_df[-indexes, ]

##############################################################################################
### PCA  ###
##############################################################################################
# Selecting numeric variables
numericVars <- train_df %>% select(where(is.numeric))

# Standardizing the data
preProcValues <- preProcess(numericVars, method = c("center", "scale"))
numericVarsStd <- predict(preProcValues, numericVars)

# PCA
pcaResults <- prcomp(numericVarsStd)
summary(pcaResults)
screeplot(pcaResults, type="lines")
plot(cumsum(pcaResults$sdev^2 / sum(pcaResults$sdev^2)), xlab = "Number of Components", ylab = "Cumulative Explained Variance", type = 'b')

# Eigenvalues 
eigenvalues <- pcaResults$sdev^2

# Kaiser Criterion
componentsToRetain <- sum(eigenvalues > 1)

cat("Number of components to retain based on Kaiser Criterion:", componentsToRetain, "\n")

##############################################################################################
### Combining Principal Components with Factor Variables  ###
##############################################################################################
# Selecting the principal components
pcaData <- as.data.frame(pcaResults$x[, 1:componentsToRetain])

# Selecting the factor variables from the original dataframe
factorVars <- select(train_df, where(is.factor))

# Combining the principal components with the factor variables
combinedData <- cbind(pcaData, factorVars)

##############################################################################################
### Logistic Regression with Combined Data  ###
##############################################################################################
combinedData$average_cost_min <- train_df$average_cost_min

# Fitting the logistic regression model
logitModel <- glm(target ~ ., data = combinedData, family = binomial(link = "logit"))

# Summary of the logistic regression model
summary(logitModel)

# VALIDATION -----------------------------------------------------------------------------------
# Predicting probabilities on the validation set
validation_data$combinedData <- predict(preProcValues, select(validation_data, where(is.numeric)))
validation_pca <- predict(pcaResults, validation_data$combinedData)
validation_combinedData <- cbind(as.data.frame(validation_pca[, 1:componentsToRetain]), 
                                 select(validation_data, where(is.factor)))

validation_combinedData$average_cost_min <- validation_data$average_cost_min

# Predicting probabilities on the validation data
validation_combinedData$probabilities <- predict(logitModel, newdata = validation_combinedData, type = "response")

# AUC on validation set for those below 0.5 in 'average_cost_min'
auc_low <- roc(validation_data$target[validation_combinedData$average_cost_min < 0.5] ~ 
                 validation_combinedData$probabilities[validation_combinedData$average_cost_min < 0.5])
cat("AUC for average_cost_min < 0.5:", auc_low$auc, "\n")

# Optimizing decision threshold => trade-off between sensitivity and specificity
coords_opt <- coords(auc_low, "best", ret = c("threshold", "sensitivity", "specificity"))
# Optimal threshold
threshold_opt <- as.numeric(coords_opt["threshold"])

validation_combinedData$probabilities <- as.numeric(validation_combinedData$probabilities)

# Applying the optimal threshold to the probabilities to obtain the predicted class
validation_combinedData$predicted_class <- ifelse(validation_combinedData$probabilities > threshold_opt, 1, 0)

# Calculating the AUC for the segment with 'average_cost_min' >= 0.5 
auc_high <- roc(validation_data$target[validation_combinedData$average_cost_min >= 0.5] ~ 
                  validation_combinedData$probabilities[validation_combinedData$average_cost_min >= 0.5])
cat("AUC for average_cost_min >= 0.5:", auc_high$auc, "\n")

validation_combinedData$predicted_class <- factor(validation_combinedData$predicted_class, levels = c(0, 1))
validation_data$target <- factor(validation_data$target, levels = levels(validation_combinedData$predicted_class))

# Evaluate your classification report to see the performance
confusionMatrix(validation_combinedData$predicted_class, validation_data$target)

# Optimal thresholds--------------------------------------------------------------------------
# Subset for average_cost_min > 0.5
subset_high <- validation_combinedData$average_cost_min > 0.5

# Calculating ROC curve for this subset
roc_high <- roc(validation_data$target[subset_high] ~ validation_combinedData$probabilities[subset_high])

# Generating all coordinates (sensitivity, specificity, thresholds) from the ROC curve
coords_high <- coords(roc_high, "all", ret=c("threshold", "sensitivity", "specificity"), transpose = FALSE)

# Filtering for specificity >= 0.60
coords_high_filtered <- coords_high[coords_high$specificity >= 0.60,]

# Finding the threshold that maximizes sensitivity under the filtered conditions
optimal_index <- which.max(coords_high_filtered$sensitivity)
optimal_threshold_high <- coords_high_filtered$threshold[optimal_index]

cat("Optimal threshold for maximum sensitivity with at least 80% specificity: ", optimal_threshold_high, "\n")

# LOW
# Subset of average_cost_min < 0.5
subset_low <- validation_combinedData$average_cost_min < 0.5

# ROC curve for this subset
roc_low <- roc(validation_data$target[subset_low] ~ validation_combinedData$probabilities[subset_low])

# AUC
auc_low <- auc(roc_low)

cat("AUC for average_cost_min < 0.5: ", auc_low, "\n")

##############################################################################################
### PCA + Logistic Regression with Combined Data => Submission ###
##############################################################################################
# Data Types -------------------------------------------------------------------------------------------
test_df[categorical_vars] <- lapply(test_df[categorical_vars], as.factor)

# Feature Engineering ------------------------------------------------------------------------------------------
# Converting Connect_Date to Date type
test_df$Connect_Date <- as.Date(test_df$Connect_Date, format = "%d/%m/%y")

# Feature engineering: Extracting year, month, and tenure in days 
reference_date <- max(test_df$Connect_Date, na.rm = TRUE) + 1
test_df$Tenure_days <- as.numeric(difftime(reference_date, test_df$Connect_Date, units = "days"))
test_df$Connect_Year <- as.numeric(format(test_df$Connect_Date, "%Y"))
test_df$Connect_Month <- as.numeric(format(test_df$Connect_Date, "%m"))

# Checking for missing values  -------------------------------------------------------------------------------------------
sum(is.na(test_df))
colSums(is.na(test_df))

# Handling missing values -------------------------------------------------------------------------------------
# For numerical columns, filling missing values with the median
num_cols <- sapply(test_df, is.numeric)
test_df[, num_cols] <- lapply(test_df[, num_cols], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

test_df$Usage_Band[is.na(test_df$Usage_Band)] <- modeValue
levels(test_df$Usage_Band)[levels(test_df$Usage_Band) == ""] <- "Med"

test_df <- test_df %>%
  select(-Connect_Date)

test_numericVars <- select(test_df, where(is.numeric))
# Standardizing the test data using preProcValues from training
test_numericVarsStd <- predict(preProcValues, test_numericVars)

# Applying PCA transformation using pcaResults from training
test_pcaData <- predict(pcaResults, newdata = test_numericVarsStd)

# Combining with categorical variables 
test_factorVars <- select(test_df, where(is.factor))
test_combinedData <- cbind(test_pcaData, test_factorVars)

test_combinedData$average_cost_min <- test_df$average_cost_min

# Predicting probabilities on the test set using the logistic regression model
test_probabilities <- predict(logitModel, newdata = test_combinedData, type = "response")

submission_df <- data.frame(id = test_df$id, predicted_churn_probability = test_probabilities)

# Writes the submission dataframe to a CSV file
write.csv(submission_df, "submission_pca.csv", row.names = FALSE, quote = TRUE)

#################################
# Predicting probabilities on the test set
test_combinedData$probabilities <- predict(logitModel, newdata = test_combinedData, type = "response")

# Initialize the predicted_class column
test_combinedData$predicted_class <- NA

# Apply the threshold for average_cost_min < 0.5
subset_low <- test_combinedData$average_cost_min < 0.5
optimal_threshold_low = 0.4
test_combinedData$predicted_class[subset_low] <- ifelse(test_combinedData$probabilities[subset_low] > optimal_threshold_low, 1, 0)

# Apply the threshold for average_cost_min >= 0.5
subset_high <- test_combinedData$average_cost_min >= 0.5

optimal_threshold_high = 0.1
test_combinedData$predicted_class[subset_high] <- ifelse(test_combinedData$probabilities[subset_high] > optimal_threshold_high, 1, 0)

# Prepare the submission file
submission_df <- data.frame(id = test_df$id, predicted_churn = test_combinedData$predicted_class)

# Write to CSV
write.csv(submission_df, "final_submission.csv", row.names = FALSE, quote = TRUE)
