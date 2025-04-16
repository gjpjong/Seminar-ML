## Containt two functions to evaluate the models: with a given threshold, and with different thresholds

##### Function to evaluate the models with a given threshold #####
evaluate_rss_test <- function(rss_vec, rss_vec_delta, Y_true, model_label, Z_test, threshold) {
  n <- length(Y_true)
  
  # Scale Z_test
  scaled_Z_test <- scale(Z_test)
  scaled_Z_test[is.nan(scaled_Z_test)] <- 0
  aux_avg_test <- colMeans(scaled_Z_test)
  
  # Predictions at fixed threshold
  pred <- ifelse(rss_vec > threshold, 1, 0)
  pred_delta <- ifelse(rss_vec_delta > threshold, 1, 0)
  
  # Metrics
  TP <- sum(pred == 1 & Y_true == 1)
  FP <- sum(pred == 1 & Y_true == 0)
  TN <- sum(pred == 0 & Y_true == 0)
  FN <- sum(pred == 0 & Y_true == 1)
  
  precision <- if ((TP + FP) == 0) NA else TP / (TP + FP)
  precision <- round(precision, 3)
  accuracy  <- (TP + TN) / (TP + FP + TN + FN)
  accuracy <- round(accuracy, 3)
  fp_pct    <- 100 * FP / n
  fn_pct    <- 100 * FN / n
  
  # Imbalance 
  aux <- if (sum(pred == 1) > 0) colMeans(scaled_Z_test[pred == 1, , drop = FALSE]) else rep(0, ncol(scaled_Z_test))
  imbalance <- round(sum((aux - aux_avg_test)^2), 3)
  
  # Delta policy 
  delta_changes <- round(mean(pred != pred_delta) * 100, 1)
  
  # Output
  data.frame(
    model     = model_label,
    threshold = threshold,
    precision = precision,
    accuracy  = accuracy,
    imbalance = imbalance,
    delta_policy = delta_changes
  )
}

##### Function to evaluate the models at multiple thresholds #####
evaluate_rss_test_thresholds <- function(rss_vec, rss_vec_delta, Y_true, model_label, Z_test) {
  thresholds <- seq(0, 1, by = 0.1)
  n <- length(Y_true)
  
  scaled_Z_test <- scale(Z_test)
  scaled_Z_test[is.nan(scaled_Z_test)] <- 0
  aux_avg_test <- colMeans(scaled_Z_test)
  
  res_list <- lapply(thresholds, function(th) {
    pred <- ifelse(rss_vec > th, 1, 0)
    pred_delta <- ifelse(rss_vec_delta > th, 1, 0)
    
    # Metrics
    TP <- sum(pred == 1 & Y_true == 1)
    FP <- sum(pred == 1 & Y_true == 0)
    TN <- sum(pred == 0 & Y_true == 0)
    FN <- sum(pred == 0 & Y_true == 1)
    
    precision <- if ((TP + FP) == 0) NA else TP / (TP + FP)
    precision <- round(precision, 3)
    accuracy  <- (TP + TN) / (TP + FP + TN + FN)
    accuracy <- round(accuracy, 3)
    fp_pct    <- 100 * FP / n
    fn_pct    <- 100 * FN / n
    
    # Imbalance
    aux <- if (sum(pred == 1) > 0) colMeans(scaled_Z_test[pred == 1, , drop = FALSE]) else rep(0, ncol(scaled_Z_test))
    imbalance <- round(sum((aux - aux_avg_test)^2), 3)
    
    # Delta policy
    delta_changes <- round(mean(pred != pred_delta) * 100, 1)
    
    data.frame(
      model     = model_label,
      threshold = th,
      precision = precision,
      accuracy  = accuracy,
      imbalance = imbalance,
      delta_policy = delta_changes
    )
  })
  
  do.call(rbind, res_list)
}
