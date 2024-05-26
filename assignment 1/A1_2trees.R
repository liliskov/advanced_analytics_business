# Advanced Analytics in a Big Data World -----------------------------------------------------------------------------------
# Description: Assignment 1. - 2 decision trees
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
library(caret)
library(rpart)
library(pROC)
library(ggplot2) 
library(boot)
library(glmnet)

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

# Further data exploration 
high_value_churners <- train_df %>%
  filter(target == 1) %>%
  arrange(desc(average_cost_min))

head(high_value_churners$average_cost_min, 20) # => 0.8622 - 0.5000

# Splitting the training data 2 two separate parts high vs lower value customers --------------------------------------------------------------------------
train_low <- train_df[train_df$average_cost_min < 0.5, ]
train_high <- train_df[train_df$average_cost_min >= 0.5, ]

range(train_low$average_cost_min)
range(train_high$average_cost_min)

# Splitting Data into Training and Validation Sets ------------------------------------------------------------------------------------------
set.seed(200) 
indexes <- createDataPartition(y = train_low$target, p = 0.9, list = FALSE)
train_low_data <- train_df[indexes, ]
validation_low_data <- train_df[-indexes, ]

##############################################################################################
### Decision Tree  ###
##############################################################################################
### LOW ### ----------------------------------------------------------------------------------
decision_tree_model <- rpart(target ~ ., data = train_low_data, method = "class")
summary(decision_tree_model)

predicted_dt <- predict(decision_tree_model, newdata = validation_low_data, type = "class")

confusionMatrix(predicted_dt, as.factor(validation_low_data$target))

predicted_probs <- predict(decision_tree_model, newdata = validation_low_data, type = "prob")[,2]
auc_result <- roc(response = as.numeric(validation_low_data$target)-1, predictor = predicted_probs)
print(auc_result$auc)

### Parameter Tuning ### 
best_cp <- NULL
best_auc <- 0

# Range of cp values to try
cp_values <- seq(0.001, 0.1, by = 0.001)

for (cp in cp_values) {
  model <- rpart(target ~ ., data = train_low_data, method = "class", control = rpart.control(cp = cp))
  predicted_probs <- predict(model, newdata = validation_low_data, type = "prob")[,2]
  auc_result <- roc(response = as.numeric(validation_low_data$target)-1, predictor = predicted_probs)
  auc <- auc_result$auc
  if (auc > best_auc) {
    best_auc <- auc
    best_cp <- cp
  }
}

# Best parameters:
cat("Best cp:", best_cp, "\n")
cat("Best AUC:", best_auc, "\n")

# Refitting the model using the best parameters:
best_model <- rpart(target ~ ., data = train_low_data, method = "class", control = rpart.control(cp = best_cp))

### HIGH ### ----------------------------------------------------------------------------------
# Regular
#decision_tree_model_high <- rpart(target ~ ., data = train_high, method = "class")
#summary(decision_tree_model_high)

# Adjusting weights - so that it makes the cost of false negatives higher than the cost of false positives.
# This example uses a loss matrix that increases the penalty for false negatives (predicting non-churn when it is actually churn).
decision_tree_model_high <- rpart(target ~ ., 
                                  data = train_high, 
                                  method = "class", 
                                  parms = list(loss = matrix(c(0, 1, 12, 0), nrow = 2)),
                                  control = rpart.control(cp = best_cp))

##############################################################################################
### Applying to test set  ###
##############################################################################################
# Data Types Test Data -------------------------------------------------------------------------------------------
test_df[categorical_vars] <- lapply(test_df[categorical_vars], as.factor)

# Missing Data ----------------------------------------------------------------------------------------------------------
# For numerical columns, filling missing values with the median
num_cols <- sapply(test_df, is.numeric)
test_df[, num_cols] <- lapply(test_df[, num_cols], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# For factors, filling missing values with the mode
test_df$Usage_Band[is.na(test_df$Usage_Band)] <- modeValue
levels(test_df$Usage_Band)[levels(test_df$Usage_Band) == ""] <- "Med"

sum(is.na(test_df))

# Feature Engineering Test Data ------------------------------------------------------------------------------------------
# Converting Connect_Date to Date type
test_df$Connect_Date <- as.Date(test_df$Connect_Date, format = "%d/%m/%y")

# Feature engineering: Extracting year, month, and tenure in days 
reference_date <- max(test_df$Connect_Date, na.rm = TRUE) + 1
test_df$Tenure_days <- as.numeric(difftime(reference_date, test_df$Connect_Date, units = "days"))
test_df$Connect_Year <- as.numeric(format(test_df$Connect_Date, "%Y"))
test_df$Connect_Month <- as.numeric(format(test_df$Connect_Date, "%m"))

test_df <- test_df %>%
  select(-Connect_Date)

### PREDICTIONS ##################################################################################
# Splitting the test set into high and low based on avg_cost_min
test_high <- test_df[test_df$average_cost_min >= 0.5, ]
test_low <- test_df[test_df$average_cost_min < 0.5, ]

# Predictions for each subset
# For high value customers
test_predicted_high <- predict(decision_tree_model_high, newdata = test_high, type = "prob")[,2]
# Adjusted threshold here to optimize sensitivity
# threshold <- 0.3  # Lower threshold to increase sensitivity
# test_high$prediction <- ifelse(test_predicted_high > threshold, 1, 0)
test_high$prediction <- test_predicted_high

# For low value customers
test_predicted_low <- predict(best_model, newdata = test_low, type = "prob")[,2]
# test_low$prediction <- ifelse(test_predicted_low > 0.5, 1, 0) # Assuming default threshold of 0.5 for low spenders
test_low$prediction <- test_predicted_low

# Combining the datasets back together
test_df_final <- rbind(test_high, test_low)

# Arrange by id to match original order
test_df_final <- test_df_final[order(test_df_final$id),]

### CSV ##################################################################################
# Selecting id and the prediction column
submission_df <- test_df_final %>%
  select(id, prediction)

# Write to CSV
write.csv(submission_df, "submission_2model.csv", row.names = FALSE, quote = TRUE)

# 9th place 92.26% AUC & Score: 3.643041








