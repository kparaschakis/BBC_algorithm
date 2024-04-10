# Replicate a function that can't be installed in the current version of R

copula_pobs <- function (x, na.last = "keep", ties.method = eval(formals(rank)$ties.method), 
          lower.tail = TRUE) 
{
  ties.method <- match.arg(ties.method)
  U <- if (!is.null(dim(x))) 
    apply(x, 2, rank, na.last = na.last, ties.method = ties.method)/(nrow(x) + 
                                                                       1)
  else rank(x, na.last = na.last, ties.method = ties.method)/(length(x) + 
                                                                1)
  if (inherits(x, "zoo")) 
    attributes(U) <- attributes(x)
  if (lower.tail) 
    U
  else 1 - U
}

# This function implements the multiplicity-adjusted bootstrap tilting 
# lower confidence bound estimation as described in Rink & Brannath (2022)

MabtCi <- function(
    boot, # bootstrap object as from the boot::boot function
    alpha, # significance level
    measure, # either 'class' for prediction accuracy or 'auc'
    select_id, # row# of final selected model in 'preds'
    preds, # matrix of class predictions
    cls_thresh, # classification probability threshold, e.g. 0.5
    labs # true class labels
) {
  
  unif_transfd <- copula_pobs(boot$t, ties.method = "max")
  .MaxEcdf <- unif_transfd %>% apply(1, max) %>% (stats::ecdf)
  
  t0 <- boot$t0[select_id]
  t <- boot$t[, select_id]
  
  select_preds <- preds[, select_id]
  if (measure == "class") {
    if (data.table::between(select_preds, 0, 1, incbounds = FALSE) %>% min) {
      select_preds <- ((select_preds > cls_thresh) * 1.0 == labs) * 1.0
    } else {
      select_preds <- (select_preds == labs) * 1.0
    }
  }
  
  tau_range <- switch(measure, class = c(-10, 0), auc = c(-20, 0))
  
  .EstPval <- function(.tau) {
    if (is.na(.tau)) {
      p <- NA
      tilt_weights <- NA
    } else {
      eivs <- switch(measure, 
                     class = select_preds, 
                     auc = boot::empinf(boot, index = select_id)) %>% as.matrix
      tilt_weights <- boot::exp.tilt(eivs, lambda = .tau * nrow(eivs))$p
      imp_weights <- rep(
        tilt_weights * length(select_preds), boot$R)^t(boot::boot.array(boot)) %>% 
        apply(2, prod)
      tilt_perform <- spatstat.geom::ewcdf(t, imp_weights)(t0)
      p <- 1 - .MaxEcdf(tilt_perform)
    }
    return(list(
      tau = .tau, p = p, weights = tilt_weights, success = !is.na(p) * 1.0))
  }
  
  .CalibTau <- function(.tau) {
    p <- .EstPval(.tau)$p
    obj <- (p - alpha) + 1000 * (p > alpha)  # conservative estimation
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
        bound_boot <- boot$data[, select_id] %>% 
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
  while (min_p > alpha/2 & min_tau > tau_range[1]) {
    min_tau <- min_tau - 0.1
    min_p <- .EstPval(min_tau)$p
  }
  tau_range[1] <- min_tau
  
  max_p <- .EstPval(tau_range[2])$p
  feasible <- ifelse(min_p < alpha/2 & max_p > 2*alpha, 1, 0)
  
  tau <- ifelse(feasible, stats::uniroot(.CalibTau, tau_range)$root, NA)
  tilt <- .EstPval(tau)
  bound <- .EstBound(tilt)
  
  return(bound)
}
