## This code creates all figures and results until post-processing
################ Setup ###########################

# # -- How in install BEAT
#devtools::install_version("RcppEigen", "0.3.3.7.0") ## beat does not work with newer RcppEigen
#devtools::install_version("RcppArmadillo", "0.11.4.0.1") ## beat does not work with newer RcppArmadillo
#devtools::install_github("ayeletis/beat")  ## do not update the RcppEigen if prompted
# # --


library(beat)         # For balanced_regression_forest(), etc.
library(grf)          # For regression_forest(), get_forest_weights(), etc.
library(data.table)
library(ggpubr)
library(gridExtra)
library(fastDummies)
library(readxl)
library(tidytext)
library(ggplot2)
library(Matrix)
library(foreach)
library(doParallel)
library(xgboost)
library(pROC)
library(ggplot2)
library(glmnet)  
library(readxl)
library(pROC)
library(caret)
library(ggrepel)

# Get data from kaggle page 
# https://www.kaggle.com/datasets/danofer/compass/data
excel_file <- "C:/Users/gebruiker/Documents/BEAT/cox-violent-parsed.xlsx"
df <- read_excel(excel_file)

# Change directory to where you saved this GitHub map 
dir <- "C:/Users/gebruiker/Documents/Seminar_R"
setwd(dir)

# Choose a random seed
my_seed = 42
set.seed(my_seed)

source("f_XGBoost_penalty.R")
source("f_evaluation_models.R")

######## Data Preparation & Cleaning #############

# Define the columns we want to keep
columns_to_keep <- c(
  "sex",
  "age_cat",
  "race",
  "c_charge_degree",
  "is_recid",
  "c_jail_in",
  "c_jail_out",
  "juv_fel_count",
  "juv_misd_count",
  "juv_other_count",
  "decile_score...12",
  "priors_count...15"
)

# Keep only needed columns
usefullData <- df[, columns_to_keep]

# Remove rows where 'is_recid' is unavailable
usefullData <- subset(usefullData, is_recid != -1)

data <- dummy_cols(
  usefullData,
  select_columns = c("sex", "age_cat", "race", "c_charge_degree"),
  remove_first_dummy = FALSE
)

if ("c_charge_degree_NA" %in% names(data)) {
  data <- subset(data, c_charge_degree_NA != 1)
}

data$custodyTimeServed <- ifelse(
  is.na(data$c_jail_in) | is.na(data$c_jail_out),
  0,  # Default value if either date is missing
  as.numeric(as.Date(data$c_jail_out) - as.Date(data$c_jail_in))
)

columns_to_drop_after_dummy <- c(
  "sex", "age_cat", "race",
  "c_jail_in", "c_jail_out",
  "c_charge_degree",
  "c_charge_degree_NA",
  "sex_Female"
)
dataFinal <- data[, !(names(data) %in% columns_to_drop_after_dummy)]

round(cor(dataFinal),
      digits = 2 # rounded to 2 decimals
)

Y <- dataFinal$is_recid
dataFinal <- dataFinal[, names(dataFinal) != "is_recid"]

exclude_cols <- c(
  "sex_Male",
  "age_cat_25 - 45",
  "age_cat_Greater than 45",
  "age_cat_Less than 25",
  "race_Asian",
  "race_Caucasian",
  "race_Hispanic",
  "race_Native American",
  "race_African-American",
  "race_Other"
)
X <- dataFinal[, !(names(dataFinal) %in% exclude_cols)]
Z <- dataFinal[, names(dataFinal) %in% exclude_cols]

######## Data Splitting ########
n_total <-nrow(X)
n_train <- 13116  # adjust if needed
n_test  <- n_total - n_train

X_train <- X[1:n_train, , drop = FALSE]
X_test  <- X[(n_train + 1):n_total, , drop = FALSE]
Z_train <- Z[1:n_train, , drop = FALSE]
Z_test  <- Z[(n_train + 1):n_total, , drop = FALSE]

Y_train <- as.vector(unlist(Y[1:n_train]))
Y_test  <- as.vector(unlist(Y[(n_train + 1):n_total]))

cat("Dimensions:\n")
cat("X_train:", dim(X_train), "\n")
cat("Y_train length:", length(Y_train), "\n")
cat("Z_train:", dim(Z_train), "\n")


X_train_noZ <- as.matrix(X_train)
X_test_noZ  <- as.matrix(X_test)
X_train_all <- cbind(X_train_noZ, Z_train)
X_test_all  <- cbind(X_test_noZ,  Z_test)

include_cols <- c(
  "sex_Male",
  "age_cat_Less than 25",
  "race_African-American"
)

unpriv_train <- Z_train[, names(Z_train) %in% include_cols]

unpriv <- unpriv_train

unpriv_test <- Z_test[, names(Z_test) %in% include_cols]

####### Fit Regression Forests #######
num_trees  <- 250  # adjust as needed
my_penalty <- 5
nrounds <- 200


# (1) Regression_BEAT using balanced_regression_forest
fit_regression_beat <- balanced_regression_forest(
  X_train_noZ, Y_train,
  target.weights = as.matrix(Z_train),  # using all columns of Z_train as weights
  target.weight.penalty = my_penalty,
  num.trees = num_trees
)

# (2) Regression_GRF_noZ (standard regression forest ignoring Z)
fit_regression_grf_noZ <- regression_forest(X_train_noZ, Y_train, num.trees = num_trees)

# (3) Regression_GRF_all (standard regression forest using all features)
fit_regression_grf_all <- regression_forest(X_train_all, Y_train, num.trees = num_trees)

####### XGBoost #######
fit_xgboost_penalty <- Xgboost_imbalance_penalty(
  X_train = X_train_noZ,
  Y_train = Y_train,
  Z_train = unpriv_train,
  X_test = X_test_noZ,
  lambda_pen = my_penalty,
  nrounds = num_trees,
  params = list(eval_metric = "rmse")
)

fit_xgboost <- fit_xgboost_penalty$model

######### Logistic Regression #########
train_data_log <- cbind(X_train_all, is_recid = Y_train)
log_model <- glm(is_recid ~ ., data = train_data_log, family = binomial())

####### Underlying relationship analysis  #############
## Creates the figures found in the Data section of the paper
source("c_underlying_relationship_analysis.R")

######## Variable Importance per Model ########
## Creates figure of the variable importance of the different models found in the Results section of the paper
source("c_variable_importance.R")

###### In-Sample XGBoost with penalty  #######
# Create dtrain for in-sample predictions
dtrain <- xgb.DMatrix(data = X_train_noZ, label = Y_train)

# --- Majority vote similarity scores (XGBoost) ---
train_leaf_indices <- predict(fit_xgboost, dtrain, predleaf = TRUE)
ntrees <- ncol(train_leaf_indices)
recid_indices <- which(Y_train == 1)

majority_leaf_train <- apply(train_leaf_indices, 2, function(tree_leafs) {
  tree_leafs_recid <- tree_leafs[recid_indices]
  if (length(tree_leafs_recid) == 0) return(NA)
  as.integer(names(sort(table(tree_leafs_recid), decreasing = TRUE)[1]))
})

valid_tree_idx <- which(!is.na(majority_leaf_train))
if (length(valid_tree_idx) == 0) stop("No valid trees with recidivist leaves!")

majority_leaf_train_valid <- majority_leaf_train[valid_tree_idx]
vote_counts_train <- apply(train_leaf_indices[, valid_tree_idx, drop = FALSE], 1, function(test_leafs) {
  sum(test_leafs == majority_leaf_train_valid, na.rm = TRUE)
})
similarity_scores_train <- vote_counts_train / length(valid_tree_idx)

# --- Logistic probabilities from raw margins ---
raw_preds_train <- predict(fit_xgboost, dtrain)
prob_predictions_train <- 1 / (1 + exp(-raw_preds_train))

# Out-of-sample predictions
similarity_scores_test <- fit_xgboost_penalty$get_majority_vote_similarity()
raw_preds_test <- predict(fit_xgboost, xgb.DMatrix(data = X_test_noZ))
prob_predictions_test <- 1 / (1 + exp(-raw_preds_test))

cat("Head of in-sample XGBoost majority vote similarity scores:\n")
print(head(similarity_scores_train))
cat("Head of in-sample XGBoost logistic probabilities:\n")
print(head(prob_predictions_train))

########## Using BEAT weights on a logistic model #############
Wmat_train <- tryCatch(
  get_forest_weights(fit_regression_beat),
  error = function(e) { return(NULL) }
)
if (is.null(Wmat_train)) stop("Could not extract BEAT weights for training set")
Wmat_train <- Wmat_train / rowSums(Wmat_train)  # Normalize

# Fit logistic regression using BEAT weights as input features
logistic_rss_model <- cv.glmnet(
  x = Wmat_train,
  y = Y_train,
  family = "binomial",
  alpha = 0.5,
  nfolds = 5
)

# In-sample predictions
rss_logistic_train <- predict(logistic_rss_model, newx = Wmat_train, type = "response", s = "lambda.min")

# Extract BEAT weights for test set
Wmat_test <- tryCatch(
  get_forest_weights(fit_regression_beat, newdata = X_test_noZ),
  error = function(e) { return(NULL) }
)
if (is.null(Wmat_test)) stop("Could not extract BEAT weights for test set")
Wmat_test <- Wmat_test / rowSums(Wmat_test)

# Out-of-sample predictions
rss_logistic_test <- predict(logistic_rss_model, newx = Wmat_test, type = "response", s = "lambda.min")

###### Evaluate ######
regression_grf_noZ_test <- predict(fit_regression_grf_noZ, X_test_noZ)$predictions
regression_grf_all_test <- predict(fit_regression_grf_all, X_test_all)$predictions
regression_beat_test    <- predict(fit_regression_beat, X_test_noZ)$predictions

## Log model
predicted_prob_log <- predict(log_model, newdata = X_test_all, type = "response")

## Set all NA's to 0
predicted_prob_full_na <- predicted_prob_log
predicted_prob_full_na[is.na(predicted_prob_full_na)] <- 0
predicted_class_full <- ifelse(predicted_prob_full_na > 0.5, 1, 0)

## Swap in order to get delta policy 
Z_test_delta <- Z_test
Z_test_delta <- as.data.frame(Z_test_delta)

aa_to_caucasian <- Z_test_delta$"race_African-American" == 1
# African-American → Caucasian
Z_test_delta$"race_Caucasian"[aa_to_caucasian] <- 1
Z_test_delta$"race_African-American"[aa_to_caucasian] <- 0

other_races <- !aa_to_caucasian
race_cols <- grep("^race_", names(Z_test_delta), value = TRUE)
Z_test_delta[other_races, race_cols] <- 0
Z_test_delta$"race_African-American"[other_races] <- 1

# Verify exactly one race is 1 for each individual
stopifnot(all(rowSums(Z_test_delta[, grep("^race_", names(Z_test_delta))]) == 1))

X_test_all_delta <- cbind(X_test_noZ, Z_test_delta)

# Generate predictions for the delta cases
regression_grf_noZ_delta <- predict(fit_regression_grf_noZ, X_test_noZ)$predictions  
regression_grf_all_delta <- predict(fit_regression_grf_all, X_test_all_delta)$predictions
regression_beat_delta    <- predict(fit_regression_beat, X_test_noZ)$predictions  

# Swap in order to get delta policy
Z_train_delta <- Z_train
Z_train_delta <- as.data.frame(Z_train_delta)

aa_to_caucasian <- Z_train_delta$"race_African-American" == 1

# African-American → Caucasian
Z_train_delta$"race_Caucasian"[aa_to_caucasian] <- 1
Z_train_delta$"race_African-American"[aa_to_caucasian] <- 0

other_races <- !aa_to_caucasian
race_cols <- grep("^race_", names(Z_train_delta), value = TRUE)
Z_train_delta[other_races, race_cols] <- 0
Z_train_delta$"race_African-American"[other_races] <- 1

# Verify exactly one race is 1 for each individual
stopifnot(all(rowSums(Z_train_delta[, grep("^race_", names(Z_train_delta))]) == 1))

X_train_all_delta <- cbind(X_train_noZ, Z_train_delta)

# XGBoost Probabilities
xgboost_delta <- Xgboost_imbalance_penalty(
  X_train = X_train_noZ,
  Y_train = Y_train,
  Z_train = unpriv,
  X_test = X_test_noZ,
  lambda_pen = my_penalty,
  nrounds = num_trees,
  params = list(eval_metric = "rmse")
)

fit_xgboost_delta <- xgboost_delta$model

raw_preds_delta <- predict(fit_xgboost_delta, xgb.DMatrix(data = X_test_noZ))
prob_predictions_delta <- 1 / (1 + exp(-raw_preds_test))

rss_logistic_test_delta <- predict(logistic_rss_model, newx = Wmat_test, type = "response", s = "lambda.min")

predicted_prob_full_delta <- predict(log_model, newdata = X_test_all_delta, type = "response")
predicted_prob_full_na_delta <- predicted_prob_full_delta
predicted_prob_full_na_delta[is.na(predicted_prob_full_na_delta)] <- 0
predicted_class_full_delta <- ifelse(predicted_prob_full_na_delta > 0.5, 1, 0)

# ---- Collect Results with threshold = 0.5  ----
results_df_test <- bind_rows(
  evaluate_rss_test(regression_beat_test, regression_beat_delta, Y_test, "BEAT", Z_test, 0.5),
  evaluate_rss_test(regression_grf_noZ_test, regression_grf_noZ_delta, Y_test, "GRF_noZ", Z_test, 0.5),
  evaluate_rss_test(regression_grf_all_test, regression_grf_all_delta, Y_test, "GRF_all", Z_test, 0.5),
  evaluate_rss_test(prob_predictions_test, prob_predictions_delta, Y_test, "XGBoost_BEAT_penalty", Z_test, 0.5),
  evaluate_rss_test(rss_logistic_test, rss_logistic_test_delta, Y_test, "BEAT_logistic", Z_test, 0.5),
  evaluate_rss_test(predicted_prob_full_na, predicted_prob_full_na_delta, Y_test, "Logistic", Z_test, 0.5)
)

write.csv(results_df_test, "results_all.csv", row.names = FALSE)

# ---- Collect Results for multiple thresholds ----
results_df_test_thresholds <- bind_rows(
  evaluate_rss_test_thresholds(regression_beat_test, regression_beat_delta, Y_test, "BEAT", Z_test),
  evaluate_rss_test_thresholds(regression_grf_noZ_test, regression_grf_noZ_delta, Y_test, "GRF_noZ", Z_test),
  evaluate_rss_test_thresholds(regression_grf_all_test, regression_grf_all_delta, Y_test, "GRF_all", Z_test),
  evaluate_rss_test_thresholds(prob_predictions_test, prob_predictions_delta, Y_test, "XGBoost_BEAT_penalty", Z_test),
  evaluate_rss_test_thresholds(rss_logistic_test, rss_logistic_test_delta, Y_test, "BEAT_logistic", Z_test),
  evaluate_rss_test_thresholds(predicted_prob_full_na, predicted_prob_full_na_delta, Y_test, "Logistic", Z_test)
)

## Save the best threshold: best = highest accuracy
best_thresholds_df <- results_df_test_thresholds %>%
  group_by(model) %>%
  slice_max(order_by = accuracy, n = 1, with_ties = FALSE) %>%
  ungroup()

## Optional: use the best thresholds for different models 
# write.csv(best_thresholds_df, "results_best_thresholds.csv", row.names = FALSE)

################ Make csv file for post-processing ##########################

test_predictions_for_csv <- data.frame(
  row_index                   = (n_train + 1):n_total,
  true_value                  = Y_test,
  regression_beat             = regression_beat_test,
  regression_grf_noZ          = regression_grf_noZ_test,
  regression_grf_all          = regression_grf_all_test,
  regression_xgboost     = regression_xgboost_prob_test, 
  regression_beat_log = rss_logistic_test, 
  regression_logistic = predicted_prob_full_na
)

test_predictions_for_csv <- test_predictions_for_csv %>% rename(regression_beat_log = `lambda.min`)

test_preds_with_Z <- cbind(test_predictions_for_csv, Z_test)

write.csv(test_preds_with_Z, "predictions.csv", row.names = FALSE)

################ Make csv file with binary predictions, threshold = 0.5 ##########

predicted_class_beat <- ifelse(regression_beat_test > 0.5, 1, 0)
predicted_class_grf_noZ <- ifelse(regression_grf_noZ_test > 0.5, 1, 0)
predicted_class_grf_all <- ifelse(regression_grf_all_test > 0.5, 1, 0)
predicted_class_xgboost <- ifelse(regression_xgboost_prob_test > 0.5, 1, 0)
predicted_class_beat_log <- ifelse(rss_logistic_test > 0.5, 1, 0)
predicted_class_log <- ifelse(predicted_prob_full_na > 0.5, 1, 0)

test_predictions_class <- data.frame(
  row_index                   = (n_train + 1):n_total,
  true_value                  = Y_test,
  regression_beat             = predicted_class_beat,
  regression_grf_noZ          = predicted_class_grf_noZ,
  regression_grf_all          = predicted_class_grf_all,
  regression_xgboost     = predicted_class_xgboost, 
  regression_beat_log = predicted_class_beat_log, 
  regression_logistic = predicted_class_log
)

test_predictions_class <- test_predictions_class %>% rename(regression_beat_log = `lambda.min`)

test_preds_class_with_Z <- cbind(test_predictions_class, Z_test)

write.csv(test_preds_class_with_Z, "predictions_class.csv", row.names = FALSE)

###### Make density plots ######
Z_test_plot <- as.data.frame(Z_test)

## Choose which column to use for Z (in this case, column 5 = African_American)
i <- 5 
group_name <- names(Z_test_plot)[i]

test_data <- data.frame(
  true_reg = Y_test,
  regression_beat = regression_beat_test,
  regression_grf_noZ = regression_grf_noZ_test,
  regression_grf_all = regression_grf_all_test,
  regression_xgboost = prob_predictions_test,
  regression_log_beat <- rss_logistic_test,
  regression_log <- predicted_prob_full_na,
  Z = as.factor(Z_test_plot[[i]])
)

## Rename columns
names(test_data)[names(test_data) == 'lambda.min'] <- "regression_log_beat"
names(test_data)[names(test_data) == 'regression_log....predicted_prob_full_na'] <- "regression_log"


dat.plot <- as.data.table(test_data)

p1_reg <- ggdensity(
  data    = dat.plot,
  x       = "true_reg",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "True Regression (Y)"
)

p2_reg <- ggdensity(
  data    = dat.plot,
  x       = "regression_beat",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "BEAT"
)

p3_reg <- ggdensity(
  data    = dat.plot,
  x       = "regression_grf_noZ",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "GRF_noZ"
)

p4_reg <- ggdensity(
  data    = dat.plot,
  x       = "regression_grf_all",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "GRF_all"
)

p_reg <- ggarrange(p1_reg, p2_reg, p3_reg, p4_reg, ncol = 4)

# Display the first plot
print(p_reg)

setwd(dir)

ggsave("Figure_density_plots.jpg",p_reg, width = 16, height = 6)

p5_reg <- ggdensity(
  data    = dat.plot,
  x       = "regression_xgboost",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "XGBoost_BEAT_penalty"
)

p6_reg <- ggdensity(
  data    = dat.plot,
  x       = "regression_log_beat",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "BEAT_logistic"
)

p7_reg <- ggdensity(
  data    = dat.plot,
  x       = "regression_log",
  color   = "Z",
  fill    = "Z",
  alpha   = 0.2,
  add     = "mean",
  title   = "Logistic"
)

# Collect all plots into one figure
p_reg <- ggarrange(p1_reg, p2_reg, p3_reg, p4_reg, p5_reg, p6_reg, p7_reg, ncol = 4, nrow = 2)
print(p_reg)

setwd(dir)
ggsave("Figure_density_plots.jpg",p_reg, width = 16, height = 12)

#################### ROC and AUC ####################
test_predictions <- data.frame(
  row_index                   = (n_train + 1):n_total,
  true_value                  = Y_test,
  regression_beat = regression_beat_test,
  regression_grf_noZ = regression_grf_noZ_test,
  regression_grf_all = regression_grf_all_test,
  regression_xgboost = prob_predictions_test,
  regression_log_beat = rss_logistic_test,
  regression_log = predicted_prob_full_na,
  sample                      = "out-of-sample"
)

roc_beat         <- roc(test_predictions$true_value, test_predictions$regression_beat)
roc_grf_noZ      <- roc(test_predictions$true_value, test_predictions$regression_grf_noZ)
roc_grf_all      <- roc(test_predictions$true_value, test_predictions$regression_grf_all)
roc_xgb_prob     <- roc(test_predictions$true_value, test_predictions$regression_xgboost)
roc_log_beat <- roc(test_predictions$true_value, test_predictions$lambda.min)
roc_log <- roc(test_predictions$true_value, test_predictions$regression_log)

setwd(dir)

jpeg("Figure_ROC_and_AUC.jpg", width = 1000, height = 600)

plot(roc_beat, col = "blue", legacy.axes = TRUE)
lines(roc_grf_noZ, col = "red")
lines(roc_grf_all, col = "green")
lines(roc_xgb_prob, col = "orange")
lines(roc_log_beat, col = "purple")
lines(roc_log, col = "pink")

legend("bottomright", legend = c(
  paste("BEAT (AUC =", round(auc(roc_beat), 3), ")"),
  paste("GRF_noZ (AUC =", round(auc(roc_grf_noZ), 3), ")"),
  paste("GRF_all (AUC =", round(auc(roc_grf_all), 3), ")"),
  paste("XGBoost_BEAT_penalty  (AUC =", round(auc(roc_xgb_prob), 3), ")"),
  paste("BEAT_logistic (AUC =", round(auc(roc_log_beat), 3), ")"),
  paste("Logistic (AUC =", round(auc(roc_log), 3), ")")
), col = c("blue", "red", "green", "orange", "purple", "pink"), lwd = 2)

dev.off()


#### Imbalance accuracy trade-off: different lambda's ####
## Make the figure that shows the trade-off betweeen accuracy and imbalance for different lambda values (3,5,8)
source("c_trade_off.R")
