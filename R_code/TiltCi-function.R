
TiltCi <- function(boot, alpha, measure, preds, cls_thresh, labs) {
  
  t0 <- boot$t0
  t <- boot$t
  
  select_preds <- preds
  if (measure == "class") {
    select_preds <- ((select_preds > cls_thresh) * 1.0 == labs) * 1.0
  }
  
  tau_range <- switch(measure, class = c(-10, 0), auc = c(-20, 0))
  
  .EstPval <- function(.tau) {
    if (is.na(.tau)) {
      p <- NA
      tilt_weights <- NA
    } else {
      eivs <- switch(measure, 
                     class = select_preds, 
                     auc = boot::empinf(boot)) %>% as.matrix
      tilt_weights <- boot::exp.tilt(eivs, lambda = .tau * nrow(eivs))$p
      imp_weights <- rep(tilt_weights * length(labs), boot$R)^t(boot::boot.array(boot)) %>% apply(2, prod)
      p <- (sum(imp_weights * (t > t0)) + 
              sum(imp_weights * (t == t0)) / 2) / length(t)
    }
    return(list(
      tau = .tau, p = p, weights = tilt_weights, success = !is.na(p) * 1.0))
  }
  
  .CalibTau <- function(.tau) {
    p <- .EstPval(.tau)$p
    obj <- (p - alpha) + 1000 * (p > alpha)
    return(obj)
  }
  
  .EstBound <- function(.tilt) {
    if (! .tilt$success) {
      bound <- NA
    } else {
      .weights <- .tilt$weights
      if (measure == "class") {
        bound <- weighted.mean(select_preds, .weights)
      }
      if (measure == "auc") {
        bound_boot <- boot$data %>% 
          boot::boot(function(d, i) {
            pROC::auc(labs[i], d[i], quiet=TRUE)}, length(t), stype = "i", 
            strata = labs, weights = .weights)
        bound <- mean(bound_boot$t)
      }
    }
    return(bound)
  }
  
  min_tau <- 0  
  min_p <- 1
  while (min_p > alpha / 2 & min_tau > tau_range[1]) {
    min_tau <- min_tau - 1
    min_p <- .EstPval(min_tau)$p
  }
  tau_range[1] <- min_tau
  
  max_p <- .EstPval(tau_range[2])$p
  feasible <- ifelse(min_p < alpha / 2 & max_p > 2*alpha, 1, 0)
  
  tau <- ifelse(feasible, stats::uniroot(.CalibTau, tau_range)$root, NA)
  tilt <- .EstPval(tau)
  bound <- .EstBound(tilt)
  
  return(bound)
}
