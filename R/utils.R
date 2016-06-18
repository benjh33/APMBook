#' Print model details
model_details <- function(model) {
  data("model_specs", envir = environment())
  Filter(function(m) { m[[2]] == model}, model_specs)
}

