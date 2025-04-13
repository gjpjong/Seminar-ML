####### Method to perform the Underlying relationship analysis  #############

## functions to run univariate regressions
columns_to_keep_var <- c(
  "sex",
  "age",
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
data_var_use <- df[, columns_to_keep_var]

# Remove rows where 'is_recid' is unavailable (marked by -1)
data_var_use <- subset(data_var_use, is_recid != -1)

data_var <- dummy_cols(
  data_var_use,
  select_columns = c("sex", "age_cat", "race", "c_charge_degree"),
  remove_first_dummy = FALSE
)

if ("c_charge_degree_NA" %in% names(data)) {
  data_var <- subset(data_var, c_charge_degree_NA != 1)
}

data_var$custodyTimeServed <- as.numeric(as.Date(data_var$c_jail_out) - as.Date(data_var$c_jail_in))

columns_to_drop_after_dummy_var <- c(
  "sex", "age_cat", "race",
  "c_jail_in", "c_jail_out",
  "c_charge_degree",
  "c_charge_degree_NA",
  "sex_Female"
)
data_var <- data_var[, !(names(data_var) %in% columns_to_drop_after_dummy_var)]

# Extract the outcome vector (Y). Assuming the outcome column is "is_recid"
Y <- data_var[, "is_recid"]

setDT(data_var)

feature_keys = c("juv_fel_count", 
                 "decile_score...12",
                 "juv_misd_count", 
                 "juv_other_count",  
                 "priors_count...15", 
                 'custodyTimeServed',
                 "c_charge_degree_(CO3)",
                 "c_charge_degree_(CT)",
                 "c_charge_degree_(F1)",
                 "c_charge_degree_(F2)",
                 "c_charge_degree_(F3)",
                 "c_charge_degree_(F5)",
                 "c_charge_degree_(F6)",
                 "c_charge_degree_(F7)",
                 "c_charge_degree_(M1)",
                 "c_charge_degree_(M2)",
                 "c_charge_degree_(MO3)",
                 "c_charge_degree_(NI0)",
                 "c_charge_degree_(TCX)",
                 "c_charge_degree_(X)"
) # total 20
feature_cols = lapply(feature_keys, function(x)grep(x, names(data_var), value=TRUE, fixed = TRUE))
feature_cols = unlist(feature_cols)   #5

data_var[, nonMale := 1-as.integer(sex_Male==1)]
data_var[, nonWhite := 1- as.integer(race_Caucasian==1)]
data_var[, Age_gt31 := age>median(age)]

data_reg_var = data_var[, .SD,.SDcols=c(feature_cols, 'nonMale', "nonWhite", "age","Age_gt31", "is_recid")]


## OLS for Age
ols = function(formula, data){
  out = lm(formula = formula, 
           data=data,
           na.action = na.omit)
  return(out)
}

## Logit for nonMale, nonWhite
glm_logit = function(formula, data){
  out = glm(formula = formula, 
            data=data, 
            family="binomial", 
            na.action = na.omit, 
            control=list(maxit=1000))
  return(out)
}

# Collect regression coefficients
collect_regression_coefs = function(Y, X, data, reg_fn){
  fm = as.formula(paste0(Y, "~", paste0(sprintf("`%s`", X), collapse = "+")))
  res = reg_fn(fm, data)
  res_summary = summary(res)
  coef = as.data.table(res_summary$coefficients, keep.rownames = TRUE)
  #  coef = coef[!grepl(paste0("^",factor_cols, collapse = "|"),rn)]
  coef = coef[rn!='(Intercept)']
  coef[, rn:=gsub("`", '', rn)]
  coef[, chunk:= tstrsplit(rn, "_", keep=1)]
  
  setorder(coef, Estimate)
  setnames(coef, "Std. Error", "Std")
  names(coef)[5] = "Prob"
  #  coef[, significant_005 := ifelse(Prob<0.05, "True", "False")]
  coef[, `p-val<0.05` := ifelse(Prob<0.05, "True", "False")]
  coef = coef[,.(rn, Estimate,
                 Std =  Std , 
                 dep_var = Y,
                 `p-val<0.05`,
                 chunk
  )]
  return(coef)
}

compile_coefs = function(Y, reg_fn){
  data = data_reg_var
  coef <- do.call(rbind, lapply(feature_cols, function(x) collect_regression_coefs(Y=Y, X=x, data, reg_fn=reg_fn)))
  
  coef[, chunk_len:=.N, by="chunk"]
  coef[chunk_len==1, chunk:= "Misc."]
  setorder(coef, Estimate)
  return(coef)
}

## Collect regression coefficients for Age, nonMale, nonWhite
coef_age <- compile_coefs(Y="age", reg_fn=ols)

coef_nonmale<- compile_coefs(Y="nonMale", reg_fn=glm_logit)

coef_nonwhite <- compile_coefs(Y="nonWhite", reg_fn=glm_logit)

## Now also for Y
coef_recid <- compile_coefs(Y = "is_recid", reg_fn = glm_logit)


## Plot features
all_features <- rbind(
  coef_age[, .SD, .SDcols = names(coef_age)],
  coef_nonmale[, .SD, .SDcols = names(coef_nonmale)],
  coef_nonwhite[, .SD, .SDcols = names(coef_nonwhite)]
)

all_features$dep_var = factor(all_features$dep_var, levels=c("age","nonMale", "nonWhite"))
levels(all_features$dep_var) <- c("age","Non-Male","Non-White")

## Add this line to make the plot 
all_features <- all_features %>% rename(pval_sign = `p-val<0.05`)

features_Y <- coef_recid[, .SD, .SDcols = names(coef_recid)]
features_Y$dep_var = factor(features_Y$dep_var, levels = "is_recid")
levels(features_Y$dep_var) <- "Recid"
features_Y <- features_Y %>% rename(pval_sign = `p-val<0.05`)

feature_labels <- data.table(
  rn = feature_cols, 
  label = c("juv_fel", "decile_scores", "juv_misd", "juv_other", "priors", "Time in custody", "CO3",
            "CT",
            "F1",
            "F2",
            "F3",
            "F5",
            "F6",
            "F7",
            "M1",
            "M2",
            "MO3",
            "NI0",
            "TCX",
            "X")  
)

# Merge labels into all_features
all_features <- merge(all_features, feature_labels, by = "rn", all.x = TRUE)

features_Y <- merge(features_Y, feature_labels, by = "rn", all.x = TRUE)

p_features=ggbarplot(all_features, 
                     x="label", y="Estimate", 
                     color="pval_sign", 
                     fill = "pval_sign",
                     facet.by = "dep_var",
                     scales='free',
                     ncol=3,
                     add = "point",
                     add.params = list(color="black")
) +
  coord_flip() + 
  geom_hline(yintercept = 0)+ 
  geom_errorbar(mapping=aes(ymin = Estimate - Std, 
                            ymax = Estimate + Std))+ theme(plot.title = element_text(hjust = 0.5)) +
  xlab("Estimates") +  theme(axis.title.y=element_blank())
print(p_features)

setwd(dir)
ggsave("Figure_Significance.jpg",p_features, width = 10, height = 6)

p_features_y=ggbarplot(features_Y, 
                       x="label", y="Estimate", 
                       color="pval_sign", 
                       fill = "pval_sign",
                       facet.by = "dep_var",
                       scales='free',
                       ncol=3,
                       add = "point",
                       add.params = list(color="black")) +
  coord_flip() + 
  geom_hline(yintercept = 0)+ 
  geom_errorbar(mapping=aes(ymin = Estimate - Std, 
                            ymax = Estimate + Std))+ theme(plot.title = element_text(hjust = 0.5)) +
  xlab("Estimates") +  theme(axis.title.y=element_blank())
print(p_features_y)

setwd(dir)
ggsave("Figure_Significance_Y.jpg", p_features_y, width = 6, height = 6)


######## without charge degrees #########
all_features_nocharge <- rbind(
  coef_age[, .SD, .SDcols = names(coef_age)],
  coef_nonmale[, .SD, .SDcols = names(coef_nonmale)],
  coef_nonwhite[, .SD, .SDcols = names(coef_nonwhite)]
)

all_features_nocharge <- all_features_nocharge[rn %in% c("juv_fel_count", "decile_score...12", "juv_misd_count", "juv_other_count", "priors_count...15", "custodyTimeServed")]

all_features_nocharge$dep_var = factor(all_features_nocharge$dep_var, levels=c("age","nonMale", "nonWhite"))
levels(all_features_nocharge$dep_var) <- c("age","Non-Male","Non-White")

all_features_nocharge <- all_features_nocharge %>% rename(pval_sign = `p-val<0.05`)

features_Y_nocharge <- coef_recid[, .SD, .SDcols = names(coef_recid)]
features_Y_nocharge <- features_Y_nocharge[rn %in% c("juv_fel_count", "decile_score...12", "juv_misd_count", "juv_other_count", "priors_count...15", "custodyTimeServed")]
features_Y_nocharge$dep_var = factor(features_Y_nocharge$dep_var, levels = "is_recid")
levels(features_Y_nocharge$dep_var) <- "Recid"
features_Y_nocharge <- features_Y_nocharge %>% rename(pval_sign = `p-val<0.05`)

feature_labels_nocharge <- data.table(
  rn = c("juv_fel_count",   
         "decile_score...12",
         "juv_misd_count", 
         "juv_other_count",  
         "priors_count...15",
         'custodyTimeServed'), 
  label = c("juv_fel", "decile_scores", "juv_misd", "juv_other", "priors", "Time in custody")  
)

# Merge labels into all_features
all_features_nocharge <- merge(all_features_nocharge, feature_labels_nocharge, by = "rn", all.x = TRUE)

features_Y_nocharge = merge(features_Y_nocharge, feature_labels_nocharge, by = "rn", all.x = TRUE)

p_features_nocharge=ggbarplot(all_features_nocharge, 
                              x="label", y="Estimate", 
                              color="darkturquoise", 
                              fill = "darkturquoise",
                              facet.by = "dep_var",
                              scales='free',
                              ncol=3,
                              add = "point",
                              add.params = list(color="black")) +
  coord_flip() + 
  geom_hline(yintercept = 0)+ 
  geom_errorbar(mapping=aes(ymin = Estimate - Std, 
                            ymax = Estimate + Std))+ theme(plot.title = element_text(hjust = 0.5)) +
  xlab("Estimates") +  theme(axis.title.y=element_blank())
print(p_features_nocharge)

setwd(dir)
ggsave("Figure_Significance_nocharge.jpg",p_features_nocharge, width = 10, height = 6)

p_features_nocharge_Y=ggbarplot(features_Y_nocharge, 
                                x="label", y="Estimate", 
                                color="darkturquoise", 
                                fill = "darkturquoise",
                                facet.by = "dep_var",
                                scales='free',
                                ncol=3,
                                add = "point",
                                add.params = list(color="black")) +
  coord_flip() + 
  geom_hline(yintercept = 0)+ 
  geom_errorbar(mapping=aes(ymin = Estimate - Std, 
                            ymax = Estimate + Std))+ theme(plot.title = element_text(hjust = 0.5)) +
  xlab("Estimates") +  theme(axis.title.y=element_blank())
print(p_features_nocharge_Y)

setwd(dir)
ggsave("Figure_Significance_nocharge_Y.jpg", p_features_nocharge_Y, width = 6, height = 6)

print('All figures succesfully created')
