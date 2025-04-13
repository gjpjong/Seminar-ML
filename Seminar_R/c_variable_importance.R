###### Method to make figure of the variable importance ######
# Compute variable importances for each model
vi_beat <- variable_importance(fit_regression_beat)
vi_noZ  <- variable_importance(fit_regression_grf_noZ)
vi_all  <- variable_importance(fit_regression_grf_all)
vi_xgboost_beat <- xgb.importance(model = fit_xgboost_penalty$model)
vi_log <- varImp(log_model)
## To better visualise: take logs of all values
log_importance <- log1p(vi_log$Overall)

# Convert them to data frames with column names
df_vi_beat <- data.frame(
  variable   = colnames(X_train_noZ),
  importance = vi_beat,
  method     = "BEAT"
)

df_vi_noZ <- data.frame(
  variable   = colnames(X_train_noZ),
  importance = vi_noZ,
  method     = "GRF_noZ"
)

df_vi_all <- data.frame(
  variable   = colnames(X_train_all),
  importance = vi_all,
  method     = "GRF_all"
)

df_vi_xgb_beat <- vi_xgboost_beat
df_vi_xgb_beat$method <- "XGBoost_BEAT_penalty"


df_vi_log <- data.frame(
  variable = rownames(vi_log),
  importance = log_importance, 
  method = "Logistic"
)

## Sort importance from highest to lowest for all dataframes
df_vi_beat <- df_vi_beat %>%
  arrange(desc(importance)) 

setDT(df_vi_beat)

df_vi_all <- df_vi_all %>%
  arrange(desc(importance))  

setDT(df_vi_all)

df_vi_noZ <- df_vi_noZ %>%
  arrange(desc(importance)) 

setDT(df_vi_noZ)


df_vi_xgb_beat <- df_vi_xgb_beat %>%
  arrange(desc(Gain))  

setDT(df_vi_xgb_beat)

df_vi_log <- df_vi_log %>%
  arrange(desc(importance)) 

setDT(df_vi_log)

print('Variable importance succesfully calculated')

# Combine all into one data frame
TOP_N=10

# Create the faceted bar plot (with highest importance at the top in each facet)

p1 = ggbarplot(df_vi_beat[1:TOP_N], 
               x="variable", 
               y='importance', 
               fill="variable", 
               color='variable', 
               xlab="",
               ylab="",
               legend='none',
               title="BEAT") + 
  coord_flip() + 
  scale_x_discrete(limits=rev)+ 
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 16))
print(p1)

p2 = ggbarplot(df_vi_all[1:TOP_N], 
               x="variable", 
               y='importance', 
               fill="variable", 
               color='variable',
               xlab="",
               ylab="",
               legend='none',
               title="GRF_all") + 
  coord_flip() + 
  scale_x_discrete(limits=rev)+ 
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 16))


p3 =  ggbarplot(df_vi_noZ[1:TOP_N], 
                x="variable", 
                y='importance', 
                fill="variable", 
                color='variable', 
                xlab="",
                ylab="",
                legend='none',
                title="GRF_noZ") + 
  coord_flip() + 
  scale_x_discrete(limits=rev)+ 
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 16))


p4 =  ggbarplot(df_vi_xgb_beat[1:TOP_N], 
                x="Feature", 
                y='Gain', 
                fill="Feature", 
                color='Feature', 
                xlab="",
                ylab="",
                legend='none',
                title="XGB_BEAT_penalty") + 
  coord_flip() + 
  scale_x_discrete(limits=rev) + 
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 16))

p5 = ggbarplot(df_vi_log[1:TOP_N], 
               x="variable", 
               y='importance', 
               fill="variable", 
               color='variable', 
               xlab="",
               ylab="",
               legend='none',
               title="Logistic") + 
  coord_flip() + 
  scale_x_discrete(limits=rev) + 
  theme(axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 16))
print(p5)


p_vi <- ggarrange(p1,p2,p3,p4,p5, ncol = 5)
print(p_vi)           

ggsave("Figure_Variable_Importance_with_Log.jpg", p_vi, width = 30, height = 16)

print('Figure of variable importance succesfully created')