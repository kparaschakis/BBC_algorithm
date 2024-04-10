source("MabtCi-function.R")
source("TiltCi-function.R")

# set.seed(2022)
cls_thresh <- 0
alpha <- 0.05
n_runs <- 100

datasets <- list.files('../real_datasets/JAD_results/', pattern = 'run_0_outcome')
for (d in 1:length(datasets)) datasets[d] <- strsplit(datasets[d], '_run')[[1]][1]
for (d in 1:length(datasets)){
  dataset <- datasets[d]
  if (!file.exists(paste('real_datasets_results/', dataset, '.csv', sep=''))){
    labs_0 <- read.csv(paste('../real_datasets/JAD_results/', dataset, '_run_0_outcome.csv', sep=''))[, 'X0']
    if (length(unique(labs_0)) == 2){
      winners_valid <- c()
      winners_eval <- c()
      for (run in 0:(n_runs - 1)){
        labs <- read.csv(paste('../real_datasets/JAD_results/', dataset, '_run_', run, '_outcome.csv', sep=''))[, 'X0']
        split_indices <- read.csv(paste('../real_datasets/JAD_results/', dataset, '_run_', run, '_splitIndices.csv', sep=''))
        folds <- rep(NA, length(labs))
        for (i in 1:length(folds)) folds[i] <- which(rowSums(split_indices == i-1, na.rm=T) == 1) - 1
        n_folds <- length(unique(folds))
        preds <- read.csv(paste('../real_datasets/JAD_results/', dataset, '_run_', run, '_predictions.csv', sep=''))
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
        
        unique_folds <- sort(unique(folds))
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
        winners_valid <- c(winners_valid, best_model)
        top_10p_models <- which(valid_performs >= quantile(valid_performs, 0.9))
        
        eval_aucs <- apply(eval_preds, 2, function(.preds) {
          pROC::auc(eval_labs, .preds, quiet=TRUE)}) %>% unname
        
        final_model_10p <- eval_aucs[top_10p_models] %>% which.max
        winners_eval <- c(winners_eval, top_10p_models[final_model_10p])
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
        cat('Dataset:', d, 'of', length(datasets), '/ run:', run + 1, 'of', n_runs, '\n')
      }
      # Comparison of results ----
      summaries <- cbind(winners_valid, delong_best_df, hanley_best_df, bt_best_df, winners_eval, delong_10p_df, hanley_10p_df, bt_10p_df, mabt_10p_df)
      write.csv(summaries, paste('real_datasets_results/', dataset, '.csv', sep=''))
    }
  }
}

#