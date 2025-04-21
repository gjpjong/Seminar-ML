#########################################
##### Code for XGBoost with penalty #####
#########################################

Xgboost_imbalance_penalty <- function(X_train, Y_train, Z_train, X_test,
                                      lambda_pen = 10, nrounds = 100,
                                      params = list(eval_metric = "rmse")) {
  # Convert training and test predictors to matrices.
  X_train <- as.matrix(X_train)
  X_test  <- as.matrix(X_test)
  
  # Compute a composite summary for the protected attribute(s) from Z_train.
  z_train <- rowMeans(as.matrix(Z_train))  # or replace with direct factor if binary
  
  # Create an XGBoost DMatrix from the training data and attach the protected summary.
  dtrain <- xgb.DMatrix(data = X_train, label = Y_train)
  attr(dtrain, "z") <- z_train  # Attach z_train as an attribute.
  
  # Define a custom objective function using logistic loss and a continuous penalty term.
  custom_obj <- function(preds, dtrain) {
    # Retrieve true labels and protected attribute summary for this batch.
    y <- getinfo(dtrain, "label")
    z <- attr(dtrain, "z")
    
    # Compute probabilities using the logistic function.
    p <- 1 / (1 + exp(-preds))
    
    # Improved delta: difference between average predictions for the two groups.
    zf <- factor(z)
    if (nlevels(zf) == 2) {
      grp_means <- tapply(p, zf, mean)
      delta     <- abs(diff(unname(grp_means)))
    } else {
      delta <- 0  # fallback if not exactly two groups
    }
    
    # Compute gradient and Hessian with exponential penalty form.
    grad <- (p - y) + lambda_pen * delta * (1 - 2*y) * exp((1 - 2*y) * preds)
    hess <- p * (1 - p) + lambda_pen * delta * exp((1 - 2*y) * preds)
    
    return(list(grad = grad, hess = hess))
  }
  
  # Train the XGBoost model using the custom objective.
  booster <- xgb.train(
    params  = params,
    data    = dtrain,
    nrounds = nrounds,
    obj     = custom_obj,
    verbose = 0
  )
  
  
  # Create a DMatrix for the test data.
  dtest <- xgb.DMatrix(data = X_test)
  
  # Get raw predictions on the test set.
  predictions <- predict(booster, dtest)
  
  ##############################################
  # Helper Function: get_leaf_cooccurrence
  ##############################################
  get_leaf_cooccurrence <- function() {
    
    leaf_mat <- predict(booster, dtrain, predleaf = TRUE)
    n <- nrow(leaf_mat)
    ntrees <- ncol(leaf_mat)
    
    cl <- makeCluster(detectCores() - 1)
    registerDoParallel(cl)
    
    cooc_list <- foreach(j = 1:ntrees, .packages = "Matrix") %dopar% {
      f <- factor(leaf_mat[, j])
      M_j <- sparseMatrix(i = 1:n,
                          j = as.integer(f),
                          x = 1,
                          dims = c(n, length(levels(f))))
      tcrossprod(M_j)
    }
    
    stopCluster(cl)
    cooc <- Reduce(, cooc_list)
    return(cooc)
  }
  
  ##############################################
  # Helper Function: get_majority_vote_similarity
  ##############################################
  get_majority_vote_similarity <- function() {
    train_leaf_indices <- predict(booster, dtrain, predleaf = TRUE)
    ntrees <- ncol(train_leaf_indices)
    recid_indices <- which(Y_train == 1)
    
    majority_leaf <- apply(train_leaf_indices, 2, function(tree_leafs) {
      tree_leafs_recid <- tree_leafs[recid_indices]
      if (length(tree_leafs_recid) == 0) {
        return(NA)
      }
      as.integer(names(sort(table(tree_leafs_recid), decreasing = TRUE)[1]))
    })
    
    test_leaf_indices <- predict(booster, dtest, predleaf = TRUE)
    
    vote_counts <- apply(test_leaf_indices, 1, function(test_leafs) {
      sum(test_leafs == majority_leaf, na.rm = TRUE)
    })
    
    similarity_score <- vote_counts / ntrees
    return(similarity_score)
  }
  
  ##############################################
  # Helper Function: get_scaled_predictions
  ##############################################
  get_scaled_predictions <- function() {
    raw_preds <- predict(booster, dtest)
    scaled_preds <- (raw_preds - min(raw_preds)) / (max(raw_preds) - min(raw_preds))
    return(scaled_preds)
  }
  
  # Return the booster, raw predictions, and helper functions.
  return(list(
    model = booster,
    predictions = predictions,
    get_leaf_cooccurrence = get_leaf_cooccurrence,
    get_majority_vote_similarity = get_majority_vote_similarity,
    get_scaled_predictions = get_scaled_predictions
  ))
}
