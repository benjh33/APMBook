
#' @export
`%||%` <- function(x, y) {
  if(is.null(x))
    return(y)
  x
}


#' @export
`%NF%` <- function(x, y) {
  if(is.null(x) || identical(x, FALSE)) return(y)
  x
}

#' This should be able to take a list of models which 
#' respond to 'predict' and an appropriately long
#' Y vector and facet plot pred ~ Y
plotPreds <- function(modList, Y, ...) {
  args <- list(...)
  prefix <- if('prefix' %in% names(args)) {
    args$prefix
  } else { 'pred' }
  ptitle <- class(modList[[1]])[1]
  m <- matrix(nrow = length(Y),
              ncol = length(modList) + 1)
  m[, 1] <- Y
  newd <- args$newdata %||% mod$call$data
  m[, 2:ncol(m)] <- sapply(modList, function(mod, ...) {
    pred <- tryCatch({
      mf <- model.frame(mod$call$form, 
                        data = newd)
      predict(mod, newdata = mf)
    }, error = function(e) {
      print(e)
    },
    finally = function() rep(NA, nrow(newd)))}
    ,...)
  rmses <- apply(m[, 2:ncol(m)], 2, caret::RMSE, obs = Y)
  m <- as.data.frame(m)
  names(m) <- c('endpoint', paste0(sprintf('%s_', prefix), 
                                   names(modList)))
  names(rmses) <- names(m)[2:length(m)]
  rmses <- as.data.frame(rmses)
  rmses <- rmses %>% mutate(
    variable = rownames(.),
    endpoint = diff(range(Y)) * 0.7,
    value = apply(m[,2:ncol(m)], 2, max) * 0.9
  )
  m <- reshape2::melt(m, id.vars = 'endpoint')
  ncol <- args$ncol %||% 2
  nrow <- args$nrow %||% ceiling(length(mods)/ncol)
  ggplot(m, aes(x = value, y = endpoint)) + 
    geom_point(alpha = 0.3) +
    theme_bw() + facet_wrap(~variable, 
                            ncol = ncol, nrow = nrow) + 
    geom_text(data = rmses, aes(x = value, y = endpoint, 
                    label = paste('RMSE:\n', round(rmses, 3))),
              color = 'blue', hjust = 1) +
    ggtitle(ptitle)
}

#' Observed v. Pred and resid v. pred
#' 
#' @export
obs_resid_pred <- function(df, ids = 'pred', ...) {
  melted <- reshape2::melt(df, id.vars = ids)
  gp_args <- unclass(ggplot2::GeomPoint$default_aes)
  dots <- pryr::dots(...)
  for(x in names(gp_args)) {
    gp_args[[x]] <- dots[[x]] %NF% gp_args[[x]]
  }
  ggplot2::ggplot(melted, aes(x = pred, y = value)) +
    plyr::splat(geom_point)(gp_args) +
    facet_wrap(~variable, scales = 'free') +
    theme_bw() + 
    ggtitle(list(...)$title %NF% 
              'Predicted v.\nresid and obs.') 
}

