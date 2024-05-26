# Advanced Analytics in a Big Data World -----------------------------------------------------------------------------------
# Description: Assignment 1.
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

# Splitting Data into Training and Validation Sets ------------------------------------------------------------------------------------------
set.seed(200) 
indexes <- createDataPartition(y = train_df$target, p = 0.8, list = FALSE)
train_data <- train_df[indexes, ]
validation_data <- train_df[-indexes, ]

# Piecharts --------------------------------------------------------------------------------------------------------------------------------------
for(col_name in names(train_df)) {
  if(is.factor(train_df[[col_name]]) | is.character(train_df[[col_name]])) {
    # Overall distribution
    overall_dist <- table(train_df[[col_name]])
    # Distribution when target == 1
    target_1_dist <- table(train_df[train_df$target == 1, ][[col_name]])
    # Distribution when target == 0
    target_0_dist <- table(train_df[train_df$target == 0, ][[col_name]])
    # Plot layout
    par(mfrow = c(1, 3))
    # Plots
    pie(overall_dist, main = paste("Overall Distribution of", col_name))
    pie(target_1_dist, main = paste("Distribution of", col_name, "\n(Target == 1)"))
    pie(target_0_dist, main = paste("Distribution of", col_name, "\n(Target == 0)"))
    cat("Press <Enter> to continue with the next variable:\n")
    readline()
  }
}

# Histograms --------------------------------------------------------------------------------------------------------------------------------------
for(col_name in names(train_df)) {
  if(is.numeric(train_df[[col_name]])) {
    hist(train_df[[col_name]], main=paste("Histogram of", col_name), xlab=col_name)
    cat("Press <Enter> to continue:\n")
    readline()
  }
}

##############################################################################################
### Decision Tree  ###
##############################################################################################
decision_tree_model <- rpart(target ~ ., data = train_data, method = "class")
summary(decision_tree_model)

predicted_dt <- predict(decision_tree_model, newdata = validation_data, type = "class")

confusionMatrix(predicted_dt, as.factor(validation_data$target))

predicted_probs <- predict(decision_tree_model, newdata = validation_data, type = "prob")[,2]
auc_result <- roc(response = as.numeric(validation_data$target)-1, predictor = predicted_probs)
print(auc_result$auc)

##############################################################################################
### Tree on high spenders only  ###
##############################################################################################
high_spenders_threshold <- quantile(train_data$`average cost min`, 0.92)
high_spenders_data <- train_data[train_data$`average cost min` > high_spenders_threshold, ]
high_spenders_model <- rpart(target ~ ., data = high_spenders_data, method = "class")

validation_high_spenders <- validation_data[validation_data$`average cost min` > high_spenders_threshold, ]
predicted_high_spenders <- predict(high_spenders_model, newdata = validation_high_spenders, type = "class")

confusionMatrix(predicted_high_spenders, as.factor(validation_high_spenders$target))

predicted_probs <- predict(high_spenders_model, newdata = validation_high_spenders, type = "prob")[,2]
auc_result <- roc(response = as.numeric(validation_high_spenders$target)-1, predictor = predicted_probs)
print(auc_result$auc)

### Parameter Tuning ### ------------------------------------------------------------------------------
best_cp <- NULL
best_auc <- 0

# Range of cp values to try
cp_values <- seq(0.001, 0.1, by = 0.001)

for (cp in cp_values) {
  model <- rpart(target ~ ., data = high_spenders_data, method = "class", control = rpart.control(cp = cp))
  predicted_probs <- predict(model, newdata = validation_high_spenders, type = "prob")[,2]
  auc_result <- roc(response = as.numeric(validation_high_spenders$target)-1, predictor = predicted_probs)
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
best_model <- rpart(target ~ ., data = high_spenders_data, method = "class", control = rpart.control(cp = best_cp))

##############################################################################################
### Tree - Submission  ###
##############################################################################################
# Data Types Test Data -------------------------------------------------------------------------------------------
categorical_vars <- c("Gender", "tariff", "Handset", "Usage_Band", "Tariff_OK", "high Dropped calls", "No Usage")
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

### HIGH-SPENDERS SUBMISSION CSV ##################################################################################
test_predicted_probs_high <- predict(high_spenders_model, newdata = test_df, type = "prob")
test_df$PRED_high <- test_predicted_probs_high[,2]

submission_df_high <- test_df %>%
  select(id, PRED_high)

write.csv(submission_df_high, "submission_high.csv", row.names = FALSE, quote = TRUE)

##############################################################################################
### Logistic Regression - All Data ###
##############################################################################################
logistic_model <- glm(target ~., 
                      data = train_data, 
                      family = binomial(link = "logit"))

summary(logistic_model)

# Regularization with glmnet --------------------------------------------------------------------------------
# Preparing the data
x <- model.matrix(target ~ . - 1, data = train_data) # Creating the model matrix, -1 to exclude intercept
y <- as.numeric(train_data$target) - 1  # Convert target to binary (0 and 1)

# Fitting the Elastic Net model with 5-fold cross-validation
set.seed(123) # For reproducibility
cv_model <- cv.glmnet(x, y, family = "binomial", alpha = 1, nfolds = 5)

# Best lambda value (where the model has the lowest cross-validated error)
best_lambda <- cv_model$lambda.min

# Model coefficients at best lambda
print(coef(cv_model, s = best_lambda))

# Prediction on new data
validation_data_without_target <- validation_data %>% select(-target)
x_test <- model.matrix(~ . - 1, data = validation_data_without_target) 
predictions <- predict(cv_model, newx = x_test, s = best_lambda, type = "response")

# Converting probabilities to binary class predictions based on a threshold (e.g., 0.5)
predicted_class <- ifelse(predictions > 0.5, 1, 0)

conf_matrix <- confusionMatrix(as.factor(predicted_class), as.factor(validation_data$target))
print(conf_matrix)

roc_result <- roc(response = as.numeric(validation_data$target) - 1, predictor = as.numeric(predictions), levels = rev(levels(as.factor(validation_data$target))))
print(roc_result$auc)

##############################################################################################
### Logistic Regression - All - Submission ###
##############################################################################################
# Preparing the model matrix for test_df
test_data_without_id <- test_df %>% select(-id, -PRED_high)
x_test_df <- model.matrix(~ . - 1, data = test_data_without_id)

# Predict probabilities with the cv_model
test_predicted_probs <- predict(cv_model, newx = x_test_df, s = best_lambda, type = "response")

# Submission dataframe
submission_df <- data.frame(id = test_df$id, PRED_high = test_predicted_probs)

# Write to CSV file for submission
write.csv(submission_df, "submission_logreg.csv", row.names = FALSE, quote = TRUE)

##############################################################################################
### Logistic Regression - High Spenders ###
##############################################################################################
logistic_model <- glm(target ~., 
                      data = high_spenders_data, 
                      family = binomial(link = "logit"))

summary(logistic_model)

# Regularization with glmnet --------------------------------------------------------------------------------
# Preparing the data
x <- model.matrix(target ~ . - 1, data = high_spenders_data) # Creating the model matrix, -1 to exclude intercept
y <- as.numeric(high_spenders_data$target) - 1  # Convert target to binary (0 and 1)

# Fitting the Elastic Net model with 5-fold cross-validation
set.seed(123) # For reproducibility
cv_model <- cv.glmnet(x, y, family = "binomial", alpha = 1, nfolds = 5)

# Best lambda value (where the model has the lowest cross-validated error)
best_lambda <- cv_model$lambda.min

# Model coefficients at best lambda
print(coef(cv_model, s = best_lambda))

# Prediction on new data
validation_data_without_target <- validation_high_spenders %>% select(-target)
x_test <- model.matrix(~ . - 1, data = validation_data_without_target) 
predictions <- predict(cv_model, newx = x_test, s = best_lambda, type = "response")

# Converting probabilities to binary class predictions based on a threshold (e.g., 0.5)
predicted_class <- ifelse(predictions > 0.5, 1, 0)

conf_matrix <- confusionMatrix(as.factor(predicted_class), as.factor(validation_high_spenders$target))
print(conf_matrix)

roc_result <- roc(response = as.numeric(validation_high_spenders$target) - 1, predictor = as.numeric(predictions), levels = rev(levels(as.factor(validation_high_spenders$target))))
print(roc_result$auc)

##############################################################################################
### Logistic Regression - High Spenders - Submission ###
##############################################################################################
# Preparing the model matrix for test_df
test_data_without_id <- test_df %>% select(-id)
x_test_df <- model.matrix(~ . - 1, data = test_data_without_id)

# Predict probabilities with the cv_model
test_predicted_probs <- predict(cv_model, newx = x_test_df, s = best_lambda, type = "response")

# Submission dataframe
submission_df <- data.frame(id = test_df$id, PRED_high = test_predicted_probs)

# Write to CSV file for submission
write.csv(submission_df, "submission_high_logreg.csv", row.names = FALSE, quote = TRUE)


##############################################################################################
### Simplified Logistic Regression - High Spenders w/ interactions ###
##############################################################################################









