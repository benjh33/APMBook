library(AppliedPredictiveModeling)
library(caret)
library(e1071)
library(earth)
library(kernlab)
library(nnet)
library(doMC)
library(reshape2)
registerDoMC(cores = 4)

d1 <- data.frame(
  x = rnorm(60, 0, 2)
)
d1$y <- (d1$x * 3 - 1) + runif(60, -4, 4)
d1 <- rbind(d1, data.frame(x = -5, y = 20))
svm1 <- svm(y ~ x, d1, kernel = 'linear', epsilon = 0.5)
d1$svm <- predict(svm1)

ggplot(d1, aes(x = x, y = y)) + geom_point() + 
  geom_smooth(method = 'lm', color = 'red') + 
  geom_line(aes(y = svm), color = 'blue', size = 1) +
  theme_bw()

svm1$epsilon
lm1 <- lm(y~x, data = d1)
d1$svm_resid <- resid(svm1)
d1$lm_resid <- resid(lm1)
d1$lm_pred <- lm1$fitted.values

ggplot(d1, aes(x = svm, y = svm_resid, 
               shape = abs(svm_resid) < svm1$epsilon,
               color = abs(svm_resid) < svm1$epsilon)) +
  geom_point() + 
  scale_y_continuous(lim = c(-5, 5)) +
  theme_bw()

# page 162
## Neural Networks
data(solubility)

tooHigh <- findCorrelation(cor(solTrainXtrans), cutoff = 0.75)
trainXnnet <- solTrainXtrans[, -tooHigh]
testXnnet <- solTrainXtrans[, -tooHigh]

nnetFit <- nnet(trainXnnet, solTrainY,
                size = 5, decay = 0.01,
                linout = TRUE, trace = FALSE,
                maxit = 500,
                maxNWts = 5 * (ncol(trainXnnet) + 1) + 5 + 1)

nnetAvg <- avNNet(trainXnnet, solTrainY,
                size = 5, decay = 0.01,
                repeats = 5,
                linout = TRUE, trace = FALSE,
                maxit = 500,
                maxNWts = 5 * (ncol(trainXnnet) + 1) + 5 + 1)
preds <- data.frame(
  obs = solTestY,
  nnetAvg = predict(nnetAvg, solTestX[, -tooHigh]),
  nnet = predict(nnetFit, solTestX[, -tooHigh])
)


nnetGrid <- expand.grid(.decay = c(0, 0.01, 0.1),
                        .size = 1:10,
                        .bag = FALSE)
set.seed(100)
idx <- createFolds(solTrainY, list = TRUE, returnTrain = TRUE)
ctrl <- trainControl('cv', index = idx)
# this takes longer than 30 minutes on my machine
nnetTune <- train(trainXnnet, solTrainY,
                  method = 'avNNet',
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  preProc = c('center', 'scale'),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(trainXnnet) + 1) + 10 + 1,
                  maxit = 500)


# MARS

marsFit <- earth(solTrainXtrans, solTrainY)
summary(marsFit)
plotmo(marsFit)
marsGrid <- expand.grid(degree = 1:2, nprune = 2:38)
set.seed(100)
marsTune <- train(solTrainX, solTrainY,
                  method = 'earth', 
                  tuneGrid = marsGrid,
                  trControl = ctrl)

summary(marsTune)
marsTune
varImp(marsTune)

preds$mars <- predict(marsTune, solTestXtrans)


# SVM
## uses kernlab
svmFit <- ksvm(x = as.matrix(solTrainXtrans), y = solTrainY,
               C = 1, epsilon = 0.1)
str(svmFit)
svmFit@kernelf@kpar
# other kernels
# polydot
# vanilladot

# using resampling
svmRTuned <- train(solTrainXtrans, solTrainY,
                   method = 'svmRadial',
                   preProc = c('center', 'scale'),
                   tuneLength = 14,
                   trControl = ctrl)

summary(svmRTuned)
preds$svm_radial <- predict(svmRTuned, solTestXtrans)

# KNN

knnDesc <- solTrainXtrans[, -nearZeroVar(solTrainXtrans)]
set.seed(100)
knnTune <- train(knnDesc,
                 solTrainY,
                 method = 'knn',
                 preProc = c('center', 'scale'),
                 tuneGrid = data.frame(.k = 1:20),
                 trControl = ctrl)

# Exercises

## 7.1

set.seed(100)
x <- runif(100, min = 2, max = 10)
y <- sin(x) + rnorm(length(x)) * 0.25
sinData <- data.frame(x = x, y = y)

dataGrid <- data.frame(x = seq(2, 10, length = 100))

svmGrid <- as.matrix(expand.grid(sigma = seq(0.1, 2, length = 5),
            cost = seq(0.1, 5, length = 10),
            eps = seq(0.1, 0.5, length = 4)))
svms <- apply(svmGrid, 1, function(r) {
  ksvm(x = x, y = y, data = sinData,
       kernel = 'rbfdot', kpar = list(sigma = r[1]),
       C = r[2], epsilon = r[3])
})
preds <- lapply(svms, predict)
param_names <- apply(svmGrid, 1, function(r) {
  sprintf("sigma_%s-cost_%s-eps_%s", r[1], r[2], r[3])
})
sinData[,param_names] <- preds

sinData <- melt(sinData, id.vars = c('x', 'y'))
sinData[, c('sigma', 'cost', 'eps')] <- matrix(unlist(
  stringr::str_split(sinData$variable, '-')),
          ncol = 3, byrow = TRUE)

ggplot(sinData, aes(x = x, y = value, color = eps)) +
  geom_point(aes(y = y), alpha = 0.1, color = 'black') +
  geom_point(alpha = 0.3) +
  theme_bw() + 
  facet_grid(cost ~ sigma) 

# 7.2

library(mlbench)
set.seed(200)
trainingData <- mlbench.friedman1(200, sd = 1)
trainingData$x <- data.frame(trainingData$x)
featurePlot(trainingData$x, trainingData$y)

testData <- mlbench.friedman1(5000, sd = 1)
testData$x <- data.frame(testData$x)

# fit nn, mars, svm and knn
nnetGrid <- expand.grid(
  .decay = c(0, 0.01, 0.1),
  .size = 3:7,
  .bag = FALSE
)
idx <- createFolds(trainingData$y)
ctrl <- trainControl('cv', index = idx)
nnetTune <- train(trainingData$x, trainingData$y,
                  method = 'avNNet',
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  preProc = c('center', 'scale'),
                  linout = TRUE,
                  trace = FALSE,
                  maxit = 500)
preds <- data.frame(obs = testData$y,
                    pred_nnet = predict(nnetTune, testData$x))

# MARS
marsGrid <- expand.grid(degree = 1:2, nprune = 2:10)
marsTune <- train(trainingData$x, trainingData$y,
                  method = 'earth',
                  tuneGrid = marsGrid,
                  trControl = ctrl)
preds$pred_mars <- predict(marsTune, testData$x[, ])[, 1]
marsTune$bestTune

# SVM

svmTune <- train(trainingData$x, trainingData$y,
                  method = 'svmRadial',
                  preProc = c('center', 'scale'),
                  tuneLength = 14,
                  trControl = ctrl)

preds$pred_svm <- predict(svmTune, testData$x)
svmTune

# knn
knnTune <- train(trainingData$x,
                 trainingData$y,
                 method = 'knn',
                 preProc = c('center', 'scale'),
                 tuneGrid = data.frame(.k = 1:20),
                 trControl = ctrl)

preds$pred_knn <- predict(knnTune, testData$x)

preds <- melt(preds, id.vars = 'obs')

ggplot(preds, aes(x = value, y = obs)) +
  facet_wrap(~variable) + theme_bw() + geom_point(alpha = 0.05)

varImp(marsTune)
varImp(svmTune)
varImp(knnTune)
varImp(nnetTune)

library(dplyr)
preds %>% group_by(variable) %>%
  summarise(
    RMSE = RMSE(obs, value)
  )

# 7.3
## Tecator data
rm(list = ls())
data(tecator)

indx <- createFolds(endpoints[,2], returnTrain = TRUE)
ctrl <- trainControl('cv', index = indx)

hists <- apply(absorp, 2, function(x) {
  qplot(x = x, geom = 'histogram', bins = 50)
})
summaries <- apply(absorp, 2, function(x) {
  s <- summary(x)
  v <- var(x)
  m <- mean(x)
  list(summary = s, var = v, mean = m)
})
absorp <- data.frame(absorp)
abPreProc <- preProcess(absorp)
scaled_ab <- apply(absorp, 2, scale)
tec_cor <- cor(absorp)


absTrainX <- predict(abPreProc, absorp)
absTrainY <- endpoints[, 2]
svmTune <- train(absTrainX, absTrainY, 
                 method = 'svmRadial',
                 tuneLength = 14,
                 trControl = ctrl)

knnTune <- train(absTrainX, absTrainY,
                 method = 'knn',
                 tuneLength = 20,
                 trControl = ctrl)

marsGrid <- expand.grid(degree = 1:2, nprune = 2:38)

marsTune <- train(absTrainX, absTrainY,
                  method = 'earth',
                  tuneGrid = marsGrid,
                  trControl = ctrl)


nnetGrid <- expand.grid(decay = c(0, 0.01, 0.1),
                        size = 1:10,
                        bag = FALSE)

nnetTune <- train(absTrainX, absTrainY,
                  method = 'avNNet',
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  trControl = ctrl,
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(absTrainX) + 1) + 10 + 1,
                  maxit = 500)

nnetPCATune <- train(absTrainX, absTrainY,
                  method = 'avNNet',
                  preProcess = 'pca',
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(absTrainX) + 1) + 10 + 1,
                  maxit = 500)

# only used two comps
nnetPCATune$preProcess$numComp
nnetTune$bestTune

nnetPCATune$results
nnetTune$results

# PCA with nnet halved the RMSE to about 10.7

svmTune
# basic svm got RMSE of 3.56
marsTune
# RMSE 2.0
knnTune
# RMSE 8.7

varImp(marsTune)

# 7.4 
rm(list = ls())
data("permeability")
# run relevant ch 6 lines
nnetGrid <- expand.grid(decay = c(0.01, 0.1),
                        size = 1:10,
                        bag = FALSE)
nnetGrid$bag <- NULL
nnetTune <- train(fingerProcessed, fingerTrainY,
                  method = 'avNNet',
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(absTrainX) + 1) + 10 + 1,
                  maxit = 500)

svmTune <- train(fingerProcessed, fingerTrainY, 
                 method = 'svmRadial',
                 tuneLength = 14,
                 trControl = ctrl)
# 10.21 RMSE
svmTuneSigma <- train(fingerProcessed, fingerTrainY, 
                 method = 'svmRadialSigma',
                 tuneLength = 14,
                 trControl = ctrl)
svmPolyTune <- train(fingerProcessed, fingerTrainY, 
                     method = 'svmPoly',
                     tuneGrid = expand.grid(
                       degree = 1:2,
                       scale = 1,
                       C = c(0.5, 1, 1.5)
                     ),
                     trControl = ctrl)

marsGrid <- expand.grid(degree = 1:2, nprune = seq(2, 130, by = 5))
marsTune <- train(fingerProcessed, fingerTrainY, 
                  method = 'earth',
                  tuneGrid = marsGrid,
                  trControl = ctrl)

models <- list(
  # nnet = nnetTune,
  svm = svmTune,
  svmPoly = svmPolyTune,
  svmSigma = svmTuneSigma,
  mars = marsTune
)
preds <- lapply(models, function(model) {
  data.frame(obs = fingerTestY,
             pred = predict(model, fingerTestX))
})
# mars is weird

preds[[4]]$pred <- predict(marsTune, fingerTestX)[, 1]
defaultSummary(preds[[1]])
defaultSummary(preds[[2]])
defaultSummary(preds[[3]])
defaultSummary(preds[[4]])

# looks like non linear does not improve

# 7.5
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

bagImputer <- preProcess(processPredictors, 
                         method = c('center', 'scale', 'bagImpute'))
bagImputed <- predict(bagImputer, processPredictors)
## this does not impute all values, only missing ones.

index <- createDataPartition(yield, p = 0.8, list = FALSE)
knnTrainX <- knnImputed[index, ]
knnTestX <- knnImputed[-index, ]
knnTrainY <- yield[index]
knnTestY <- yield[-index]

nnetGrid <- expand.grid(.decay = c(0, 0.01, 0.1),
                        .size = 1:10,
                        .bag = FALSE)
indx <- createFolds(knnTrainY, returnTrain = TRUE)
ctrl <- trainControl('cv', index = indx)
nnetTune <- train(knnTrainX, knnTrainY,
                  method = 'avNNet',
                  preProcess = 'pca',
                  tuneGrid = nnetGrid,
                  trControl = ctrl)

svmTune <- train(knnTrainX, knnTrainY,
                 method = 'svmRadial',
                 tuneLength = 10,
                 trControl = ctrl
                 )
svmPolyTune <- train(knnTrainX, knnTrainY,
                     method = 'svmPoly',
                     tuneGrid = expand.grid(
                       degree = 1:2,
                       C = seq(2, 20, by = 2),
                       scale = 1:3
                     ),
                     trControl = ctrl)
marsGrid <- expand.grid(degree = 1:2, nprune = seq(2, 50, by = 2))
marsTune <- train(knnTrainX, knnTrainY,
                  method = 'earth',
                  tuneGrid = marsGrid,
                  trControl = ctrl)

knnTune <- train(knnTrainX, knnTrainY,
                 method = 'knn',
                 tuneGrid = data.frame(k = 3:20),
                 trControl = ctrl)
models <- list(
  nnet = nnetTune, 
  svm = svmTune, 
  svmPoly = svmPolyTune, 
  mars = marsTune, 
  knn = knnTune)

models

impVars <- lapply(models, varImp)
lapply(impVars, plot)
preds <- lapply(models, function(m) {
  data.frame(obs = knnTestY,
             pred = predict(m, knnTestX))
})
preds$mars$pred <- predict(marsTune, knnTestX)[, 1]
lapply(preds, defaultSummary)

svmImp <- impVars$svm$importance
svmImp <- svmImp[order(-svmImp$Overall), , drop = FALSE]

library(reshape2)
scatterData <- data.frame(yield = yield)
scatterData[ , row.names(svmImp)[1:9]] <- 
  processPredictors[, row.names(svmImp)[1:9]]
scatterData <- melt(scatterData, id.vars = 'yield')
ggplot(scatterData, aes(x = value, y = yield)) + geom_point() +
  theme_bw() + 
  facet_wrap(~variable, scales = 'free')
