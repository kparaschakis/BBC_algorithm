library(dplyr)
source("MabtCi-function.R")
source("TiltCi-function.R")

# set.seed(2022)
cls_thresh <- 0
alpha <- 0.05
n_runs <- 200

sim_confs <- list.dirs('../simulated_data_for_comparison')[-1]
for (sc in 1:length(sim_confs)){
  data_folder <- paste(sim_confs[sc], '/', sep='')
  if (!file.exists(paste('simulated_data_results/', strsplit(data_folder, '/')[[1]][3], '.csv', sep=''))){
    if (grepl('classes_2', sim_confs[sc])){
      samples <- as.numeric(strsplit(strsplit(data_folder, 'samples_')[[1]][2], '_config')[[1]][1])
      balance <- strsplit(strsplit(data_folder, 'balance_')[[1]][2], '_classes')[[1]][1]
      if (balance == 'equal') balance <- 0.5 else balance = as.numeric(balance)
      worse_case <- samples * balance * 0.75
      n_folds <- length(unique(read.csv(paste(data_folder, 'folds_0.csv', sep=''))[, 'X0']))
      if (worse_case >= n_folds){
        for (run in 0:(n_runs - 1)){
          labs <- read.csv(paste(data_folder, 'outcome_', run, '.csv', sep=''))[, 'X0']
          folds <- read.csv(paste(data_folder, 'folds_', run, '.csv', sep=''))[, 'X0']
          preds <- read.csv(paste(data_folder, 'predictions_', run, '.csv', sep=''))
          perfs <- read.csv(paste(data_folder, 'performances_', run, '.csv', sep=''))
          n_obs <- length(labs)
          
          learn_ids <- sample(1:n_obs, 0.75 * n_obs)
          eval_ids <- setdiff(1:n_obs, learn_ids)
          n_eval <- length(eval_ids)
          learn_labs <- labs[learn_ids]
          eval_labs <- labs[eval_ids]
          
          learn_folds <- folds[learn_ids]
          eval_folds <- folds[eval_ids]
          learn_preds <- preds[learn_ids, ]
          eval_preds <- preds[eval_ids, ]
          
          aggregate_1 <- aggregate(learn_labs, by=list(learn_folds), FUN=mean)
          aggregate_2 <- aggregate(eval_labs, by=list(rep(1, length(eval_labs))), FUN=mean)
          
          while ((dim(aggregate_1)[1] < n_folds) | (min(aggregate_1['x']) == 0) | (max(aggregate_1['x']) == 1) |
                 (min(aggregate_2['x']) == 0) | (max(aggregate_2['x']) == 1)){
            learn_ids <- sample(1:n_obs, 0.75 * n_obs)
            eval_ids <- setdiff(1:n_obs, learn_ids)
            n_eval <- length(eval_ids)
            learn_labs <- labs[learn_ids]
            eval_labs <- labs[eval_ids]
            
            learn_folds <- folds[learn_ids]
            eval_folds <- folds[eval_ids]
            learn_preds <- preds[learn_ids, ]
            eval_preds <- preds[eval_ids, ]
            
            aggregate_1 <- aggregate(learn_labs, by=list(learn_folds), FUN=mean)
            aggregate_2 <- aggregate(eval_labs, by=list(rep(1, length(eval_labs))), FUN=mean)
          }
          
          unique_folds <- unique(folds)
          model_fold_valid_performs <- matrix(NA, dim(preds)[2], length(unique_folds))
          for (f in 1:length(unique_folds)){
            fold_labs <- learn_labs[learn_folds == unique_folds[f]]
            for (m in 1:dim(preds)[2]){
              fold_model_preds <- learn_preds[learn_folds == unique_folds[f], m]
              model_fold_valid_performs[m, f] <- pROC::auc(fold_labs, fold_model_preds, quiet=TRUE)
            }
          }
          
          valid_performs <- model_fold_valid_performs %*% (rep(1, length(unique_folds))/length(unique_folds))
          best_model <- which.max(valid_performs)
          top_10p_models <- which(valid_performs >= quantile(valid_performs, 0.9))
          
          eval_aucs <- apply(eval_preds, 2, function(.preds) {
            pROC::auc(eval_labs, .preds, quiet=TRUE)}) %>% unname
          
          final_model_10p <- eval_aucs[top_10p_models] %>% which.max
          
          # Theoretical performance of selected model
          if (run == 0){
            theoretical_valid_df <- data.frame(theoretical_valid = perfs[best_model, 1])
            theoretical_final_df <- data.frame(theoretical_final = perfs[top_10p_models, ][final_model_10p])
          } else{
            theoretical_valid_df <- rbind(theoretical_valid_df, data.frame(theoretical_valid = perfs[best_model, 1]))
            theoretical_final_df <- rbind(theoretical_final_df,
                                          data.frame(theoretical_final = perfs[top_10p_models, ][final_model_10p]))
          }
          # standard CIs for single best model ----
          # DeLong
          delong_bound <- pROC::ci.auc(
            eval_labs, eval_preds[, best_model], conf.level = 1 - 2*alpha, quiet=TRUE)[1]
          if (run == 0){
            delong_best_df <- data.frame(DeLong = delong_bound)
          } else{
            delong_best_df <- rbind(delong_best_df, data.frame(DeLong = delong_bound))
          }
          # Hanley-McNeil
          auc <- eval_aucs[best_model]
          q1 <- auc / (2 - auc)
          q2 <- 2 * auc^2 / (1 + auc)
          n_success <- ((eval_preds[, best_model] > cls_thresh) * 1.0 == 
                          eval_labs) %>% sum 
          n_fail <- n_obs - n_success
          numerator <- auc * (1-auc) + (n_success + 1)*(q1 - auc^2) + 
            (n_fail -1)*(q2 - auc^2)
          denominator <- n_success * n_fail
          hanley_bound <- auc - qnorm(1-alpha) * sqrt(numerator / denominator)
          if (run == 0){
            hanley_best_df <- data.frame(Hanley_McNeil = hanley_bound)
          } else{
            hanley_best_df <- rbind(hanley_best_df, data.frame(Hanley_McNeil = hanley_bound))
          }
          # standard CIs for top 10% selection rule ----
          # DeLong
          sidak_alpha <- 1 - (1-alpha)^(1/length(top_10p_models))
          delong_bound <- pROC::ci.auc(
            eval_labs, eval_preds[, top_10p_models[final_model_10p]], conf.level = 1 - 2*sidak_alpha, quiet=TRUE)[1]
          if (run == 0){
            delong_10p_df <- data.frame(DeLong_10p = delong_bound)
          } else{
            delong_10p_df <- rbind(delong_10p_df, data.frame(DeLong_10p = delong_bound))
          }
          # Hanley-McNeil
          auc <- eval_aucs[top_10p_models[final_model_10p]]
          q1 <- auc / (2 - auc)
          q2 <- 2 * auc^2 / (1 + auc)
          n_success <- ((eval_preds[, top_10p_models[final_model_10p]] > cls_thresh) * 1.0 == 
                          eval_labs) %>% sum 
          n_fail <- n_obs - n_success
          numerator <- auc * (1-auc) + (n_success + 1)*(q1 - auc^2) + 
            (n_fail -1)*(q2 - auc^2)
          denominator <- n_success * n_fail
          hanley_bound <- auc - qnorm(1-alpha) * sqrt(numerator / denominator)
          if (run == 0){
            hanley_10p_df <- data.frame(Hanley_McNeil_10p = hanley_bound)
          } else{
            hanley_10p_df <- rbind(hanley_10p_df, data.frame(Hanley_McNeil_10p = hanley_bound))
          }
          # MABT CIs for top 10% selection rule ----
          aucs_boot <- boot::boot(
            eval_preds[, top_10p_models], function(d, i) {
              as.matrix(d[i, ]) %>% 
                apply(2, function(.preds) pROC::auc(eval_labs[i], .preds, quiet=TRUE))}, 2000, strata = eval_labs)
          if (run == 0){
            mabt_10p_df <- data.frame(
              mabt = MabtCi(aucs_boot, alpha, "auc", final_model_10p, eval_preds[, top_10p_models], cls_thresh, eval_labs))
          } else{
            mabt_10p_df <- rbind(mabt_10p_df,
                                 data.frame(mabt = MabtCi(aucs_boot, alpha, "auc", final_model_10p,
                                                          eval_preds[, top_10p_models], cls_thresh, eval_labs)))
          }
          # BT CIs for single best model ----
          bt_boot <- boot::boot(
            eval_preds[, best_model], function(d, i) {pROC::auc(eval_labs[i], d[i], quiet=TRUE)}, R = 2000, strata = eval_labs)
          if (run == 0){
            bt_best_df <- data.frame(
              bt = TiltCi(bt_boot, alpha, "auc", eval_preds[, best_model], cls_thresh, eval_labs))
          } else{
            bt_best_df <- rbind(bt_best_df,
                                data.frame(bt = TiltCi(bt_boot, alpha, "auc", eval_preds[, best_model], cls_thresh,
                                                       eval_labs)))
          }
          # BT CIs for top 10% selection rule ----
          sidak_alpha <- 1 - (1-alpha)^(1/length(top_10p_models))
          bt_boot <- boot::boot(
            eval_preds[, top_10p_models[final_model_10p]], 
            function(d, i) {pROC::auc(eval_labs[i], d[i], quiet=TRUE)}, R = 2000, strata = eval_labs)
          if (run == 0){
            bt_10p_df <- data.frame(bt_10p = TiltCi(bt_boot, sidak_alpha, "auc",
                                                    eval_preds[, top_10p_models[final_model_10p]],cls_thresh, eval_labs))
          } else{
            bt_10p_df <- rbind(bt_10p_df,
                               data.frame(bt_10p = TiltCi(bt_boot, sidak_alpha, "auc",
                                                          eval_preds[, top_10p_models[final_model_10p]],
                                                          cls_thresh, eval_labs)))
          }
          
          #
          cat('simulation setting:', sc, 'of', length(sim_confs), '/ run:', run + 1, 'of', n_runs, '\n')
        }
        
        # Comparison of results ----
        overall_best <- c()
        for (run in 0:(n_runs - 1)){
          overall_best <- c(overall_best, max(read.csv(paste(data_folder, 'performances_', run, '.csv', sep=''))))
        }
        summaries <- cbind(overall_best, theoretical_valid_df, delong_best_df, hanley_best_df, bt_best_df,
                           theoretical_final_df, delong_10p_df, hanley_10p_df, bt_10p_df, mabt_10p_df)
        write.csv(summaries, paste('simulated_data_results/', strsplit(data_folder, '/')[[1]][3], '.csv', sep=''))
      }
    }
  }
}

# # # Create summaries
# Read simulation configurations
simulation_results <- list.files('simulated_data_results/')
# Loop over results
summaries <- as.data.frame(matrix(NA, length(simulation_results), 18))
colnames(summaries) <- c('configuration', 'average_best',
                         'average_theoretical_validation',
                         'average_DeLong', 'PercentInclusion95_DeLong',
                         'average_HanleyMcNeil', 'PercentInclusion95_HanleyMcNeil',
                         'average_BT', 'PercentInclusion95_BT',
                         'average_theoretical_final',
                         'average_DeLong_10p', 'PercentInclusion95_DeLong_10p',
                         'average_HanleyMcNeil_10p', 'PercentInclusion95_HanleyMcNeil_10p',
                         'average_BT_10p', 'PercentInclusion95_BT_10p',
                         'average_MABT_10p', 'PercentInclusion95_MABT_10p')
for (r in 1:length(simulation_results)){
  summaries$configuration[r] <- strsplit(simulation_results[r], '.csv')[[1]][1]
  results_file <- read.csv(paste('simulated_data_results/', simulation_results[r], sep=''))
  #
  summaries$average_best[r] <- mean(results_file$overall_best)
  #
  summaries$average_theoretical_validation[r] <- mean(results_file$theoretical_valid)
  summaries$average_DeLong[r] <- mean(results_file$DeLong)
  summaries$PercentInclusion95_DeLong[r] <- mean(results_file$theoretical_valid >= results_file$DeLong)
  summaries$average_HanleyMcNeil[r] <- mean(results_file$Hanley_McNeil)
  summaries$PercentInclusion95_HanleyMcNeil[r] <- mean(results_file$theoretical_valid >= results_file$Hanley_McNeil)
  summaries$average_BT[r] <- mean(results_file$bt)
  summaries$PercentInclusion95_BT[r] <- mean(results_file$theoretical_valid >= results_file$bt)
  #
  summaries$average_theoretical_final[r] <- mean(results_file$theoretical_final)
  summaries$average_DeLong_10p[r] <- mean(results_file$DeLong_10p)
  summaries$PercentInclusion95_DeLong_10p[r] <- mean(results_file$theoretical_final >= results_file$DeLong_10p)
  summaries$average_HanleyMcNeil_10p[r] <- mean(results_file$Hanley_McNeil_10p)
  summaries$PercentInclusion95_HanleyMcNeil_10p[r] <- mean(results_file$theoretical_final >= results_file$Hanley_McNeil_10p)
  summaries$average_BT_10p[r] <- mean(results_file$bt_10p)
  summaries$PercentInclusion95_BT_10p[r] <- mean(results_file$theoretical_final >= results_file$bt_10p)
  summaries$average_MABT_10p[r] <- mean(results_file$mabt)
  summaries$PercentInclusion95_MABT_10p[r] <- mean(results_file$theoretical_final >= results_file$mabt)
}
# Store summaries
write.csv(summaries, 'summaries.csv', row.names = F)

#