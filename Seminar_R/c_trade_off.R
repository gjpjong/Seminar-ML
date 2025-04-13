#### Imbalance fairness trade-off: different lambda's ####

lambda_vals <- c(3, 5, 8)
all_models_eval_train <- list()

evaluate_at_threshold_train <- function(score_vec, Y_true, Z_input, model_name, threshold = 0.5) {
  pred <- ifelse(score_vec > threshold, 1, 0)
  acc <- mean(pred == Y_true)
  
  Z_mat <- scale(as.matrix(Z_input), center = TRUE, scale = TRUE)
  Z_mat[is.nan(Z_mat)] <- 0
  aux_avg <- colMeans(Z_mat)
  aux <- if (sum(pred == 1) > 0) colMeans(Z_mat[pred == 1, , drop = FALSE]) else rep(0, ncol(Z_mat))
  imbalance <- round(sum((aux - aux_avg)^2), 3)
  
  return(data.frame(model = model_name, threshold = threshold, accuracy = acc, imbalance = imbalance))
}

print("Running the models for lambda = [3,5,8]...")
compute_in_sample_rss <- function(fit, Y_train) {
  Wmat <- tryCatch(
    get_forest_weights(fit),
    error = function(e) { return(NULL) }
  )
  if (is.null(Wmat)) return(rep(NA, length(Y_train)))
  as.numeric(Wmat %*% Y_train)
}

for (lam in lambda_vals) {
  fit_beat <- balanced_regression_forest(
    X_train_noZ, Y_train,
    target.weights = as.matrix(Z_train),
    target.weight.penalty = lam,
    num.trees = 250
  )
  beat_scores <- compute_in_sample_rss(fit_beat, Y_train)
  all_models_eval_train[[paste0("BEAT_", lam)]] <- evaluate_at_threshold_train(beat_scores, Y_train, Z_train, paste0("BEAT_lambda_", lam))
  
  res_xgb <- Xgboost_imbalance_penalty(
    X_train = X_train_noZ,
    Y_train = Y_train,
    Z_train = Z_train,
    X_test  = X_train_noZ,
    lambda_pen = lam,
    nrounds = 200,
    params = list(eval_metric = "rmse")
  )
  xgb_probs <- 1 / (1 + exp(-predict(res_xgb$model, xgb.DMatrix(data = X_train_noZ))))
  all_models_eval_train[[paste0("XGB_", lam)]] <- evaluate_at_threshold_train(xgb_probs, Y_train, Z_train, paste0("XGBoost_lambda_", lam))
  
  Wmat_train <- tryCatch(
    get_forest_weights(fit_beat),
    error = function(e) { return(NULL) }
  )
  if (!is.null(Wmat_train)) {
    Wmat_train <- Wmat_train / rowSums(Wmat_train)
    model_rss <- cv.glmnet(x = Wmat_train, y = Y_train, family = "binomial", alpha = 0.5, nfolds = 5)
    rss_scores <- predict(model_rss, newx = Wmat_train, type = "response", s = "lambda.min")
    all_models_eval_train[[paste0("RSS_", lam)]] <- evaluate_at_threshold_train(as.numeric(rss_scores), Y_train, Z_train, paste0("RSS_Logistic_lambda_", lam))
  }
}

print("Creating the figure")

rss_grf_all <- predict(fit_regression_grf_all, X_train_all)$predictions
rss_grf_noZ <- predict(fit_regression_grf_noZ, X_train_noZ)$predictions

rss_log <- predict(log_model, newdata = X_train_all, type = "response")
rss_log[is.na(rss_log)] <- 0

all_models_eval_train[["GRF_all"]] <- evaluate_at_threshold_train(rss_grf_all, Y_train, Z_train, "GRF_all")
all_models_eval_train[["GRF_noZ"]] <- evaluate_at_threshold_train(rss_grf_noZ, Y_train, Z_train, "GRF_noZ")
all_models_eval_train[["Logistic"]] <- evaluate_at_threshold_train(rss_log, Y_train,Z_train, "Logistic")

df_lambda_models <- do.call(rbind, all_models_eval_train)
df_lambda_models$lambda <- ifelse(
  grepl("lambda_", df_lambda_models$model),
  as.numeric(gsub(".*lambda_", "", df_lambda_models$model)),
  NA
)
df_lambda_models$group <- case_when(
  grepl("^BEAT_", df_lambda_models$model) ~ "BEAT",
  grepl("^XGBoost_", df_lambda_models$model) ~ "XGBoost",
  grepl("^RSS_Logistic_", df_lambda_models$model) ~ "RSS_Logistic",
  grepl("GRF_all", df_lambda_models$model) ~ "GRF_all",
  grepl("GRF_noZ", df_lambda_models$model) ~ "GRF_noZ",
  grepl("Logistic", df_lambda_models$model) ~ "Logistic",
  TRUE ~ "Other"
)


trade_off <- ggplot(df_lambda_models, aes(x = imbalance, y = accuracy, color = group)) +
  # Lines for λ-dependent models
  geom_line(
    data = df_lambda_models[!is.na(df_lambda_models$lambda), ],
    aes(group = group),
    linewidth = 1.1,
    alpha = 0.6
  ) +
  # Points for all models
  geom_point(size = 3, stroke = 1, shape = 21, fill = "white") +
  # Text labels for λ values
  geom_text_repel(
    data = df_lambda_models[!is.na(df_lambda_models$lambda), ],
    aes(label = lambda),
    size = 4,
    nudge_y = 0.005,
    show.legend = FALSE,
    color = "black",
    bg.color = "white",
    bg.r = 0.15
  ) +
  # Labels for fixed models (GRF_all, GRF_noZ, Logistic)
  geom_text_repel(
    data = df_lambda_models[is.na(df_lambda_models$lambda), ],
    aes(label = model),
    size = 4,
    nudge_y = 0.005,
    color = "black",
    bg.color = "white",
    bg.r = 0.15,
    show.legend = FALSE
  ) +
  scale_color_brewer(palette = "Set1") +
  labs(
    x = "Imbalance",
    y = "Accuracy",
    color = "Model Group"
  ) +
  theme_minimal(base_size = 15) +
  theme(
    plot.title = element_text(face = "bold", size = 17),
    plot.subtitle = element_text(size = 13, margin = margin(b = 10)),
    legend.position = "right",
    panel.grid.major = element_blank(),  # <-- removes major gridlines
    panel.grid.minor = element_blank(),  # <-- removes minor gridlines
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.8),
    plot.background = element_rect(fill = "white", color = NA)
  )
print(trade_off)

setwd(dir)
ggsave("Figure_Trade_Off.jpg", trade_off, width = 10, height = 6)

print('Figure of the trade-off succesfully created')
