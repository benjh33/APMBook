library(AppliedPredictiveModeling)
library(caret)
library(doMC)
registerDoMC(cores = 4)

data(solubility)

solTrainFull <- cbind(solTrainX, data.frame(solubility = solTrainY))
solTrainTransFull <- cbind(solTrainXtrans, data.frame(solubility = solTrainY))
rTree <- rpart::rpart(solubility ~ ., data = solTrainFull)
# conditional tree - uses statistical tests to determine
# split instead of naive reduction in error. If the means 
# between the two splits of candidate var are not different
# up to some threshold, it's not an ideal split. RPart
# favors variables with many different unique values, 
# regardless of predictive power.
cTree <- party::ctree(solubility ~ ., data = solTrainFull)

xyplot(ctree ~ rtree, 
       data.frame(ctree = predict(cTree), rtree = predict(rTree)))

ggplot(solTrainFull, aes(x = NumCarbon, y = solubility)) +
  geom_point() + theme_bw()

ctrl <- trainControl('cv')

rpartGrid <- data.frame(cp = seq(0.0000, 0.01, length = 2))
# tune over complexity parameter
rpartTuneCP <- train(solTrainX, solTrainY, 
                     method = 'rpart',
                     tuneGrid = rpartGrid,
                     trControl = ctrl)
rpartTuneCP

# tune over depth
rpartTuneD <- train(solTrainX, solTrainY, 
                    method = 'rpart2',
                    tuneGrid = data.frame(maxdepth = 8:20),
                    trControl = ctrl)
rpartTuneD
plot(rpartTuneCP$finalModel)

## Model trees
### RWeka is complainy
library(RWeka)
# run 'sudo R CMD javareconf'
m5tree <- M5P(solubility ~ ., solTrainFull)
plot(solTrainFull$solubility, predict(m5tree))

# rules based - make initial tree, keep rule that
# has greatest coverage, remove those obs, and create new tree
# apply linear model associated with rule to get prediction
m5rules <- M5Rules(solubility ~ ., solTrainFull)
plot(solTrainFull$solubility, predict(m5rules))

m5tree <- M5P(solubility ~ ., data = solTrainFull,
              control = Weka_control(M = 10))
plot(m5tree)

# caret - doesn't work - DO NOT RUN
set.seed(100)
m5Tune <- train(solTrainXtrans, solTrainY, 
                method = 'M5', # 'M5Rules'
                trControl = trainControl('cv'),
                control = Weka_control(M = 10))

plot(m5Tune)

# bagged trees

baggedTree <- ipred::ipredbagg(solTrainY, solTrainXtrans)

baggedTree2 <- ipred::bagging(solubility ~ ., data = solTrainFull)

bagCtrl <- party::cforest_control(mtry = ncol(solTrainFull) - 1)
baggedTree3 <- party::cforest(solubility ~ ., data = solTrainFull, 
                              controls = bagCtrl)


# random forest

rfModel <- randomForest::randomForest(solTrainXtrans, solTrainY,
                                      importance = TRUE,
                                      ntree = 1000)

# or
# rfModel <- randomForest::randomForest(solubility ~ ., solTrainFull)
randomForest::importance(rfModel)
str(rfModel)
plot(rfModel)

# gbm

gbmModel <- gbm::gbm.fit(solTrainXtrans, solTrainY, 
                         distribution = 'gaussian')
# or 
# But I don't get why they used the transformed data
# should be the same.
gbmModelTrans <- gbm::gbm(solubility ~ ., data = solTrainTransFull)
gbmModelRaw <- gbm::gbm(solubility ~ ., data = solTrainFull)

# predictions are similar.
plot(predict(gbmModelTrans, n.trees = 400), predict(gbmModelRaw, n.trees = 400))

gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2),
                       .n.trees = seq(100, 1000, by = 50),
                       .shrinkage = c(0.01, 0.1),
                       .n.minobsinnode = 30)
gbmTune <- train(solTrainXtrans, solTrainY,
                 method = 'gbm',
                 tuneGrid = gbmGrid,
                 verbose = FALSE)


# Cubist

cubistMod <- Cubist::cubist(solTrainXtrans, solTrainY)
lapply(4:9, function(n) {
  RMSE(solTestY, predict(cubistMod, solTestXtrans, neighbors = n))
})

summary(cubistMod)

cubistTune <- train(solTrainXtrans, solTrainY, 
                    method = 'cubist',
                    tuneLength = 20)


## Exercises
rm(list = ls())
library(mlbench)
set.seed(200)
simulated <- mlbench.friedman1(200, sd = 1)
simulated <- cbind(simulated$x, simulated$y)
simulated <- as.data.frame(simulated)
colnames(simulated)[ncol(simulated)] <- 'y'

# a
rfModel1 <- randomForest::randomForest(y ~ ., data = simulated, 
                                     importance = TRUE, 
                                     ntree = 1000)
rfImp1 <- varImp(rfModel1, scale = FALSE)
## does the model use the uninformative predictors v6-v10?
# nope

# b
simulated$duplicate1 <- simulated$V1 + rnorm(200) * 0.1
cor(simulated$duplicate1, simulated$V1)
rfModel2 <- randomForest::randomForest(y ~ ., data = simulated, 
                                     importance = TRUE, 
                                     ntree = 1000)
rfImp2 <- varImp(rfModel2, scale = FALSE)
# does importance score for V1 change?
(rfImp2['V1', 1] - rfImp1['V1', 1])/rfImp1['V1', 1]
# droped 38%

# c 
cforestMod <- party::cforest(y ~ ., data = simulated)
tradVarImp <- party::varimp(cforestMod)
# threshold default is 0.2 - too low. Raising it
# requires X's to be more correlated with X_i in question
# to be included in conditioning grid
condVarImp <- party::varimp(cforestMod, conditional = TRUE)

varImpDF <- rbind(data.frame(value = tradVarImp, 
                             name = names(tradVarImp), 
                             type = 'unconditional'),
                  data.frame(value = condVarImp, 
                             name = names(condVarImp),
                             type = 'conditional'))
ggplot(varImpDF, aes(y = value, x = name, fill = type,
                     color = type)) +
  geom_bar(stat = 'identity', position = 'dodge') + theme_bw() + 
  coord_flip()
# variable importance on the correlated variable is 
# about half that of unconditional. Both real variables are still
# lower than the duplicated correlation

# d - do it again with other trees
## gbm
gbmModelOrig <- gbm::gbm(y ~ . - duplicate1, data = simulated,
                        dist = 'gaussian', 
                        n.trees = 500, shrinkage = 0.01)
gbmModelDup <- gbm::gbm(y ~ ., data = simulated,
                         dist = 'gaussian', 
                         n.trees = 500, shrinkage = 0.01)
gbmVarImpOrig <- gbm::relative.influence(gbmModelOrig, scale = TRUE)
gbmVarImpDup <- gbm::relative.influence(gbmModelDup, scale = TRUE)

(gbmVarImpDup['V1'] - gbmVarImpOrig['V1'])/gbmVarImpOrig['V1']
# dropped by 42% (-47% when not scaled)
gbmVarImpDup['V1'] / gbmVarImpOrig['V1']

## cubist
dep <- grep('y', names(simulated))
f <- paste(names(simulated)[-dep], collapse = '+')
f <- formula(paste("y~", f))
ffull <- update(f, ~ . - duplicate1)
cubModOrig <- Cubist::cubist(simulated[,attr(terms(ffull), 
                                             'term.labels')],
                            simulated[,'y'],
                            committees = 50)
cubModDup <- Cubist::cubist(simulated[,attr(terms(f), 'term.labels')],
                         simulated[,'y'],
                         committees = 50)

summary(cubModDup)

# indicates duplicated variable stole a lot of credit from
# V1
cubModDup$usage
cubModOrig$usage


# 2 Use a simulation to show tree bias with different granularities
rm(list = ls())
library(foreach)
## output depends on the observations used to make the tree
data(solubility)
proportionTrain <- seq(0.5, 0.9, by = 0.1)
solTrainFull <- cbind(solTrainX, data.frame(solubility = solTrainY))
ntrees <- 300
rpartMods <- foreach(prop = proportionTrain, .combine = 'rbind') %dopar% {
  rmses <- do.call('rbind', lapply(1:ntrees, function(x) {
    samp <- sample.int(nrow(solTrainX), nrow(solTrainX) * prop)
    rp <- rpart::rpart(solubility ~ ., data = solTrainFull[samp,])
    cp <- party::ctree(solubility ~ ., data = solTrainFull[samp,])
    rp_pred <- RMSE(solTestY, predict(rp, solTestX))
    cp_pred <- RMSE(solTestY, predict(cp, solTestX))
    c(rp_pred, cp_pred)
  }))
  d <- data.frame(prop = prop, rpart_rmse = rmses[, 1],
                  cpart_rmse = rmses[, 2])
  d <- reshape2::melt(d, id.vars = 'prop')
  d
}
ggplot(rpartMods, 
       aes(x = value)) + geom_histogram(bins = 45, alpha = 0.5) +
  geom_density() +
  theme_bw() +
  facet_grid(variable~prop)

# 3 Bagging fraction and learning rate in GBM

# a
# Why does high learning rate and bag fraction lead to variable
# importance being attributed to a few variables?
# - observations in models with high bagging are nearly identical
# - subsequent models depend on 90% of the outcome of the prev model
# b
# In general, lower learning rates with enough trees will predict better
# c
# How would increasing interaction depth affect the slope of predictor
# importance for either model (high bag.frac, high learning v. low)?
# as interaction depth increases, more will variables get used
library(gbm)
solTrainFull <- solTrainFull[,c(ncol(solTrainFull), 
                                1:(ncol(solTrainFull)-1))]
indx <- createFolds(solTrainY, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

testIntDepth <- function(interaction_depth = 1){
  highTune <- gbm.fit(
    solTrainXtrans, solTrainY,
    bag.fraction = 0.9,
    interaction.depth = interaction_depth,
    shrinkage = 0.9,
    distribution = 'gaussian', 
    verbose = FALSE
  )
  lowTune <- gbm.fit(
    solTrainXtrans, solTrainY,
    bag.fraction = 0.1,
    interaction.depth = interaction_depth,
    shrinkage = 0.1,
    distribution = 'gaussian',
    verbose = FALSE
  )
  varimpGbm <- merge(summary(lowTune), summary(highTune),
                     by = 'var', suffixes = c('_low', '_high'))
  varimpGbm <- reshape2::melt(varimpGbm)
  
  p <- ggplot(subset(varimpGbm, value > 1), aes(x = var, y = value)) + 
    geom_bar(stat = 'identity') + coord_flip() +
    facet_grid(~ variable) + theme_bw() 
  p + ggtitle(sprintf('int depth = %s', interaction_depth))
}

idepth_plots <- lapply(1:7, testIntDepth)
idepth_plots
# increase interaction depth sort of showed that the model with high
# tuning values exhibited greater influence from those variables
# important in the low tuning value model.

# 4
# a
rtreeModels <- lapply(seq(0.01, 0.4, length.out = 5), 
                      function(cp) {
                        rpart::rpart(solubility ~ MolWeight, 
                                     data = solTrainFull,
                            method = 'anova', cp = cp)
                      })
rforestModels <- lapply(seq(500, 1000, length = 5), 
                        function(n) {
    randomForest::randomForest(
      solTrainX[, 'MolWeight', drop = FALSE],
      solTrainY, importance = TRUE,
      ntree = n)
})
cubeTune <- expand.grid(
  committees = c(1, seq(40, 100, by = 20)),
  rules = 100
)
cubistModels <- lapply(1:nrow(cubeTune), function(row) {
  Cubist::cubist(solTrainXtrans[, c('MolWeight', 'NumCarbon'), 
                                drop = FALSE],
                 solTrainY, 
                 committees = cubeTune$committees[row],
                 control = Cubist::cubistControl(
                   rules = cubeTune$rules[row])
                 )
})
solTestDF <- cbind(solTestX[, c('MolWeight', 'NumCarbon'), 
                            drop = FALSE], solTestY)
names(solTestDF)[ncol(solTestDF)] <- 'solubility'
plotFit <- function(mods, test_data, param_fun, ...){
  id_vars <- c('solubility', 'MolWeight')
  params <- sapply(mods, param_fun)
  test_data[, params] <- lapply(mods, predict, 
                  newdata = test_data[, c('MolWeight', 'NumCarbon'), 
                                          drop = FALSE], ...)
  test_data$NumCarbon <- NULL
  test_data <- reshape2::melt(test_data, id.vars = id_vars)
  args <- list(...)
  arg_title <- paste0(sapply(names(args), function(a) { 
    paste0(a, '_', args[[a]])}), collapse = '\n')
  p <- ggplot(test_data, aes(x = MolWeight, y = value, 
                             color = variable)) +
    geom_point(aes(x = MolWeight, y = solubility), 
               inherit.aes = FALSE,
               color = 'black', alpha = 0.1) +
    geom_point(alpha = 0.3) + 
    ggtitle(paste0(class(mods[[1]]), arg_title,
            sep = '\n')) + 
    facet_wrap(~variable, nrow = 3, ncol = 3) +
    theme_bw() + 
    theme(legend.position = 'none')
    p
}
param_fun_rf <- function(model) {
  paste0('ntree_', model$ntree)
}
param_fun_rpart <- function(model) {
  paste0('cp_', model$control$cp)
}
param_fun_cubist <- function(model) {
  paste0("committees_", model$committees)
}

rfPlots <- plotFit(rforestModels, solTestDF, param_fun_rf)
rpartPlots <- plotFit(rtreeModels, solTestDF, param_fun_rpart)
cubistPlots <- plotFit(cubistModels, solTestDF, param_fun_cubist,
                       neighbors = 9)

# 5
rm(list = ls())
data(tecator)
# PCA can deal with between-predictor correlation
colnames(absorp) <- paste0('V', seq(1, ncol(absorp)))
abProcess <- preProcess(absorp, method = c('pca'), ncomp = 3)
absorpPCA <- predict(abProcess, absorp)

## rpart
cps <- seq(0, 0.4, by = 0.05)
fitRpart <- function(data, cps, formula = y ~ ., ...) {
  mods <- lapply(cps, function(x, formula, data) {
    rpart::rpart(formula, data, 
                 control = rpart::rpart.control(cp = x))
    }, 
    formula = formula,
    data = data
    )
  names(mods) <- paste('cp_', cps)
  mods
}
rpartMods <- fitRpart(cbind(data.frame(y = endpoints[,2]),
                            absorp), cps)
rpartPCAMods <- fitRpart(cbind(data.frame(y = endpoints[, 2]),
                               absorpPCA), cps)
## ctree, vary criterion
ctrls <- lapply(seq(0.75, 0.99, length = 5), function(x) {
  party::ctree_control(mincriterion = x)
})

fitCtrees <- function(data, ctrls, formula = y ~ ., ...) {
  mods <- lapply(ctrls, function(ctrl) {
    party::ctree(formula, data, controls = ctrl)
  })
  names(mods) <- paste0("mincrit_", 
                        sapply(ctrls, function(ctrl) 
                          ctrl@gtctrl@mincriterion))
  mods
}
ctreeMods <- fitCtrees(cbind(data.frame(y = endpoints[,2]),
                            absorp), ctrls)
ctreePCAMods <- fitCtrees(cbind(data.frame(y = endpoints[, 2]),
                               absorpPCA), ctrls)


## M5
fitM5 <- function(data, ctrls, formula = y ~ ., 
                  method = 'M5P', ...) {
  require(RWeka)
  mods <- lapply(ctrls, function(ctrl) {
    do.call(method, list(formula, data, control = ctrl))
  })
  names(mods) <- sapply(ctrls, `[[`, 1)
  mods
}
ctrls <- lapply(seq(2, 10, by = 2), function(m) 
  RWeka::Weka_control(M = m))

m5Mods <- fitM5(cbind(data.frame(y = endpoints[,2]),
                      absorp), ctrls)
m5PCAMods <- fitM5(cbind(data.frame(y = endpoints[,2]),
                      absorpPCA), ctrls)
plotPreds(m5Mods, endpoints[, 2], prefix = 'M')
plotPreds(m5PCAMods, endpoints[, 2], prefix = 'M')
plotPreds(ctreeMods, endpoints[, 2], prefix = '')
plotPreds(rpartMods, endpoints[, 2], prefix = '')
plotPreds(ctreePCAMods, endpoints[, 2], prefix = '')
plotPreds(rpartPCAMods, endpoints[, 2], prefix = '')

## M5Rules

m5RMods <- fitM5(cbind(data.frame(y = endpoints[,2]),
                      absorp), ctrls, method = 'M5Rules')
m5RPCAMods <- fitM5(cbind(data.frame(y = endpoints[,2]),
                         absorpPCA), ctrls, method = 'M5Rules')
plotPreds(m5RMods, endpoints[, 2])
plotPreds(m5RPCAMods, endpoints[, 2])


## Bagging
bags <- floor(seq(10, 25, length = 5))
bagMods <- lapply(bags, function(m){
  ipred::ipredbagg(endpoints[, 2], absorp, 
                   nbagg = m)
})
names(bagMods) <- bags
bagPCAMods <- lapply(bags, 
                     function(m){
  ipred::ipredbagg(endpoints[, 2], absorpPCA, 
                   nbagg = m)
})
names(bagPCAMods) <- bags
plotPreds(bagMods, endpoints[, 2])
plotPreds(bagPCAMods, endpoints[, 2])


## randomForest
mtry <- ceiling(seq(ncol(absorp)/5, ncol(absorp)/2.5,
            length.out = 5))
fitRfMods <- function(d, formula = y ~ ., ctrls, ...) {
  mods <- lapply(ctrls, function(m, data, formula, ntree) {
    randomForest::randomForest(formula, data, mtry = m, 
                               ntree = ntree)
    }, 
    formula = formula,
    data = d, ntree = 1000)
  names(mods) <- paste0('mtry_', ctrls)
  mods
}
rfMods <- fitRfMods(cbind(data.frame(y = endpoints[,2]),
                            absorp), ctrl = mtry)
  
rfPCAMods <- fitRfMods(cbind(data.frame(y = endpoints[,2]),
                       absorpPCA), ctrl = 1:ncol(absorpPCA))
plotPreds(rfMods, endpoints[,2])
plotPreds(rfPCAMods, endpoints[,2])


## cforest
fitCfMods <- function(d, formula = y ~ .,...) {
  ctrls <- seq(5, ceiling((ncol(d)-1)/3),
               length = 5)
  mods <- lapply(ctrls, function(ctrl, data, formula) {
    party::cforest(formula, data, 
                   controls = party::cforest_control(
                     ntree = 1000, 
                     mtry = ctrl)
                   )
  }, data = d, formula = formula)
  names(mods) <- paste0('mtry_', ctrls)
  mods
}
cfMods <- fitCfMods(cbind(data.frame(y = endpoints[,2]),
                             absorp))
cfPCAMods <- fitCfMods(cbind(data.frame(y = endpoints[,2]),
                             absorpPCA))
plotPreds(cfMods, endpoints[,2])
plotPreds(cfPCAMods, endpoints[,2])

## gbm
library(gbm)
help(gbm)
fractions <- seq(0.4, 0.8, by = 0.1)
gbmMods <- lapply(fractions, 
                  function(frac, ms, Ys, nts) {
  gbm.fit(ms, Ys, distribution = 'gaussian',
      bag.fraction = frac, n.trees = nts)
}, Ys = endpoints[,2], 
  ms = absorp,
  nts = 200)
gbmPCAMods <- lapply(fractions, 
                  function(frac, ms, Ys, nts, ...) {
                    gbm.fit(ms, Ys, distribution = 'gaussian',
                            bag.fraction = frac, n.trees = nts,
                            ...)
                  }, Ys = endpoints[,2], 
                  ms = absorpPCA,
                  nts = 200, verbose = FALSE)
names(gbmMods) <- fractions
names(gbmPCAMods) <- fractions
plotPreds(gbmMods, endpoints[,2], n.trees = 200)
# messed up
plotPreds(gbmPCAMods, endpoints[,2], n.trees = 200)
sapply(lapply(gbmPCAMods, predict, n.trees = 200), caret::RMSE,
       obs = endpoints[, 2])
## cubist
# etc...

# comparing these to the linear models is dumb
# with only 215 obs, test RMSE is very dependent on 
# the sampling.


# 8.6
rm(list = ls())
data("permeability")

trainIndex <- createDataPartition(permeability[, 1], p = 0.75, 
                              list = FALSE)
fingerNZV <- nearZeroVar(fingerprints, saveMetrics = TRUE)
trainY <- permeability[trainIndex, 1]
testY <- permeability[-trainIndex, 1]
trainX <- fingerprints[trainIndex, !fingerNZV$nzv]
testX <- fingerprints[-trainIndex, !fingerNZV$nzv]
# PCA is dumb here
# trainPCAProcessor <- preProcess(trainX, method = 'pca')
# trainPCAX <- predict(trainPCAProcessor, trainX)
# testPCAX <- predict(trainPCAProcessor, testX)

indx <- createFolds(trainY, k = 10, returnTrain = TRUE)
ctrl <- trainControl('cv', index = indx)

# rforest
rfTune <- train(trainX, trainY, 
                 method = 'rf',
                 trControl = ctrl,
                 tuneLength = 10)
# cforest
cfTune <- train(trainX, trainY, 
                 method = 'cforest', 
                 trControl = ctrl,
                 tuneLength = 10)
# cubist
cubeGrid <- expand.grid(
  committees = seq(10, 100, length.out = 5),
  neighbors = c(4, 7, 9)
)
cubeTune <- train(trainX, trainY, 
                  method = 'cubist',
                  trControl =  ctrl,
                  tuneGrid = cubeGrid)
# M5
library(RWeka)
library(RWekajars)
library(doMC)
m5Tuner <- expand.grid(
  pruned = c('Yes', 'No'),
  smoothed = c('Yes', 'No'),
  rules = c('Yes', 'No')
)

m5Tune <- train(trainX, trainY,
              method = 'M5',
              trControl = ctrl,
              tuneGrid = m5Tuner,
              control = Weka_control(M = 10))
m5Tuner <- expand.grid(
  pruned = c(1, 0),
  smoothed = c(1, 0),
  M = floor(seq(5, 15, length.out = 3))
)
m5Tuner <- t(as.matrix(m5Tuner))

startTime <- Sys.time()
m5Tune <- foreach(tuner = m5Tuner) %do% {
  m5ctrl <- Weka_control(M = tuner[3],
                       N = tuner[1] == 1,
                       U = tuner[2] == 1)
  mods <- lapply(ctrl$index,function(fold) {
    d <- cbind(data.frame(permeability = trainY[fold]),
               trainX[fold, ])
    mod <- M5P(permeability ~ ., d, control = m5ctrl)
    rmse <- RMSE(predict(mod, as.data.frame(trainX[-fold, ])), 
                 trainY[-fold])
    list(model = mod, rmse = rmse)
  })
  mean_rmse <- mean(sapply(mods, '[[', 'rmse'))
  list(models = mods, mean_rmse = mean_rmse)
}

endTime <- Sys.time()
endTime - startTime
results <- data.frame(
  N = vector('logical', length = length(m5Tune)),
  U = vector('logical', length = length(m5Tune)),
  M = vector('integer', length = length(m5Tune)),
  rmse = vector('integer', length = length(m5Tune))
)
for(r in 1:ncol(m5Tuner)) {
  results[r, ] <- c(m5Tuner[, r], m5Tune[[r]]$mean_rmse)
}
results[order(results$rmse), ]
results$model <- class(m5Tune[[1]]$models[[1]]$model)[1]


# M5Rules
m5RulesTune <- foreach(tuner = m5Tuner) %do% {
  m5ctrl <- Weka_control(M = tuner[3],
                         N = tuner[1] == 1,
                         U = tuner[2] == 1)
  print('control: ')
  print(m5ctrl)
  mods <- lapply(ctrl$index,function(fold) {
    d <- cbind(data.frame(permeability = trainY[fold]),
               trainX[fold, ])
    mod <- M5Rules(permeability ~ ., d, control = m5ctrl)
    rmse <- RMSE(predict(mod, as.data.frame(trainX[-fold, ])), 
                 trainY[-fold])
    list(model = mod, rmse = rmse)
  })
  mean_rmse <- mean(sapply(mods, '[[', 'rmse'))
  list(models = mods, mean_rmse = mean_rmse)
}
results_rules <- data.frame(
  N = vector('logical', length = length(m5RulesTune)),
  U = vector('logical', length = length(m5RulesTune)),
  M = vector('integer', length = length(m5RulesTune)),
  rmse = vector('integer', length = length(m5RulesTune))
)
for(r in 1:ncol(m5Tuner)) {
  results_rules[r, ] <- c(m5Tuner[, r], m5RulesTune[[r]]$mean_rmse)
}
results_rules[order(results_rules$rmse), ]
results_rules$model <- class(m5RulesTune[[1]]$models[[1]]$model)[1]

# best RMSTune
m5Mods <- rbind(results, results_rules)
forests <- list(cubeTune, rfTune, cfTune, m5RulesTune, m5Tune)
lapply(forests, function(forest) {
  if(class(forest) == 'train') {
    print(class(forest$finalModel))
    print(forest$bestTune)
    print(forest$results %>% filter(RMSE == min(RMSE)))
  } else {
    mtype <- class(forest[[1]]$models[[1]]$model)[1]
    print(mtype)
    print(m5Mods %>% filter(model == mtype) %>%
                        filter(rmse == min(rmse)))
  }
})
# random forests wins this time  mtry = 345, 
# cross-validated RMSE 10.4

# 8.7
# copied from 6_chapt.R question 6.3
rm(list = ls())
data("ChemicalManufacturingProcess")

processPredictors <- ChemicalManufacturingProcess[, -1]
yield <- ChemicalManufacturingProcess[, 1]

# use an imputation function to fill in missing values
missing <- which(sapply(processPredictors, function(col) {
  any(is.na(col))
}))

knnImputer <- preProcess(processPredictors,
                         method = 'knnImpute')
knnImputed <- predict(knnImputer, processPredictors)

indx <- createFolds(yield, returnTrain = TRUE)
ctrl <- trainControl('cv', index = indx)

# let's just use knnImputed
# rforest
rfMods <- train(knnImputed, yield, 
                method = 'rf',
                trControl = ctrl,
                tuneLength = 10)

# cforest
cfMods <- train(knnImputed, yield, 
                method = 'cforest',
                trControl = ctrl,
                tuneLength = 10)

# cubist
cubeGrid <- expand.grid(
  committees = c(1, seq(30, 100, length.out = 5)),
  neighbors = c(1, 4, 7, 9)
)
cubeTune <- train(knnImputed, yield, 
                  method = 'cubist', 
                  trControl = ctrl,
                  tuneGrid = cubeGrid)
# M5Rules
m5Grid <- expand.grid(
  pruned = c(1, 0),
  smoothed = c(1, 0)
)
m5RulesTune <- foreach(tuner = t(m5Grid)) %do% {
  weka_ctrl <- Weka_control(U = tuner[1] == 1,
                            N = tuner[2] == 1)
  mods <- lapply(ctrl$index, function(fold) {
    d <- cbind(y = yield[fold], knnImputed[fold, ])
    mod <- M5Rules(y ~ ., data = d, control = weka_ctrl)
    preds <- predict(mod, newdata = knnImputed[-fold, ])
    rmse <- RMSE(pred = preds, obs = yield[-fold])
    list(model = mod, rmse = rmse)
  })
  mean_rmse <- mean(sapply(mods, '[[', 'rmse'))
  list(models = mods, mean_rmse = mean_rmse, tune = tuner)
}

m5Grid <- expand.grid(
  pruned = c(1, 0),
  smoothed = c(1, 0),
  M = floor(seq(5, 15, length.out = 3))
)

# M5
m5Tune <- foreach(tuner = t(m5Grid)) %do% {
  weka_ctrl <- Weka_control(U = tuner[1] == 1,
                            N = tuner[2] == 1,
                            M = tuner[3])
  mods <- lapply(ctrl$index, function(fold) {
    d <- cbind(y = yield[fold], knnImputed[fold, ])
    mod <- M5P(y ~ ., data = d, control = weka_ctrl)
    preds <- predict(mod, newdata = knnImputed[-fold, ])
    rmse <- RMSE(pred = preds, obs = yield[-fold])
    list(model = mod, rmse = rmse)
  })
  mean_rmse <- mean(sapply(mods, '[[', 'rmse'))
  list(models = mods, mean_rmse = mean_rmse, tune = tuner)
}

## results
library(dplyr)
min_rmse <- min(sapply(m5RulesTune, '[[', 'mean_rmse'))
bestM5Rules <- Filter(function(tune) {
  tune$mean_rmse == min_rmse},
m5RulesTune)[[1]]
print('M5Rules')
print(bestM5Rules$tune)
print(bestM5Rules$mean_rmse)

min_rmse <- min(sapply(m5Tune, '[[', 'mean_rmse'))
bestM5 <- Filter(function(tune) {
  tune$mean_rmse == min_rmse},
  m5Tune)[[1]]
print('M5P')
print(bestM5$tune)
print(bestM5$mean_rmse)
print('random forest')
rfMods$results %>% filter(RMSE == min(RMSE))
print('conditional random forest')
cfMods$results %>% filter(RMSE == min(RMSE))
print('Cubist')
cubeTune$results %>% filter(RMSE == min(RMSE))

# cubist wins with a RMSE below 1
