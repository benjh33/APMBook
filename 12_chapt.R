rm(list = ls())
library(AppliedPredictiveModeling)
library(caret)
library(doMC)
library(dplyr)
library(ggplot2)
library(stringr)
library(glmnet)
library(MASS)
library(pamr)
library(pls)
library(rms)
library(sparseLDA)
library(subselect)
library(pROC)
registerDoMC(cores = 16)


load('data/grantData.RData')
# computing p. 308

# find extreme correlations
help(trim.matrix)
# no problems in reduced set
reducedCovMat <- cov(training[, reducedSet])
trimmingResults <- trim.matrix(reducedCovMat)
names(trimmingResults)
trimmingResults$names.discarded

# but with full set
fullCovMat <- cov(training[, fullSet])
fullSetResults <- trim.matrix(fullCovMat)
fullSetResults$names.discarded

ctrl <- trainControl(method = 'LGOCV',
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)

levels(training$Class)

# glm
set.seed(476)
modelFit <- glm(Class ~ Day, data = training[pre2008, ], 
                family = binomial)
modelFit

successProb <- 1 - predict(modelFit, newdata = data.frame(Day = c(10, 150, 300, 350)),
                           type = 'response')
ggplot(data.frame(obs = training$Class[pre2008], pred = 1 - predict(modelFit, type = 'response')),
       aes(x = obs, y = pred)) +
  geom_boxplot()

set.seed(476)
daySquaredModel <- glm(Class ~ Day + I(Day^2), 
                       data = training[pre2008, c('Class', fullSet)],
                       family = binomial)
summary(daySquaredModel)

set.seed(476)
rcsFit <- lrm(Class ~ rcs(Day), data = training[pre2008, ])
summary(rcsFit)

dayProfile <- Predict(rcsFit, Day = 0:365, fun = function(x) -x)
plot(dayProfile, ylab = "Log Odds")

training$Day2 <- training$Day^2
fullSet <- c(fullSet, 'Day2')
reducedSet <- c(reducedSet, 'Day2')
set.seed(476)
lrFull <- train(training[, fullSet], y = training$Class,
                method = 'glm', 
                metric = 'ROC', 
                trControl = ctrl)

lrFull

set.seed(476)
lrReduced <- train(training[, reducedSet], y = training$Class,
                   method = 'glm', 
                   metric = 'ROC', 
                   trControl = ctrl)
lrReduced

head(lrReduced$pred)

confusionMatrix(data = lrReduced$pred$pred, 
                reference = lrReduced$pred$obs)

reducedRoc <- roc(response = lrReduced$pred$obs, 
                  predictor = lrReduced$pred$successful,
                  levels = rev(levels(lrReduced$pred$obs)))

plot(reducedRoc)
auc(reducedRoc)

grantPreProcess <- preProcess(training[pre2008, reducedSet])

scaledPre2008 <- predict(grantPreProcess, 
                         newdata = training[pre2008, reducedSet])
scaled2008HoldOut <- predict(grantPreProcess, 
                             newdata = training[-pre2008, reducedSet])

ldaModel <- lda(x = scaledPre2008, grouping = training$Class[pre2008])

head(ldaModel$scaling)

ldaHoldOutPredictions <- predict(ldaModel, scaled2008HoldOut)
str(ldaHoldOutPredictions)
head(ldaHoldOutPredictions$posterior)

# choosing number of discriminant vectors to use is unnecessary here
# but can be done with the predict function (arg is 'dimen')
# and with train by method 'lda2' and specifying tuneLength
set.seed(476)
ldaFit1 <- train(training[, reducedSet], training$Class, 
                 method = 'lda',
                 preProc = c('center', 'scale'),
                 metric = 'ROC', trControl = ctrl)
ldaFit1
testing$Day2 <- testing$Day^2
ldaTestClasses <- predict(ldaFit1, 
                          newdata = testing[, reducedSet])
ldaTestProbs <- predict(ldaFit1, 
                          newdata = testing[, reducedSet],
                          type = 'prob')

# PLS Discriminant Analysis
plsdaModel <- plsda(training[pre2008, reducedSet], 
                    training[pre2008, 'Class'],
                    scale = TRUE,
                    probMethod = 'Bayes',
                    ncomp = 4)

plsPred <- predict(plsdaModel, 
                   newdata = training[-pre2008, reducedSet])
head(plsPred)
plsProbs <- predict(plsdaModel, 
                   newdata = training[-pre2008, reducedSet],
                   type = 'prob')
head(plsProbs)

loadings(plsdaModel)
training$Day2 <- NULL
set.seed(476)
reducedSet <- grep('Day2', reducedSet, value = TRUE, invert = TRUE)
fullSet <- grep('Day2', fullSet, value = TRUE, invert = TRUE)
plsFit2 <- train(x = training[, reducedSet],
                 y = training$Class, 
                 method = 'pls', 
                 tuneGrid = expand.grid(.ncomp = 1:10),
                 preProc = c('center', 'scale'),
                 metric = 'ROC', trControl = ctrl)
plsFit2
plsImpGrant <- varImp(plsFit2, scale = FALSE)
plot(plsImpGrant, top = 30, scales = list(y = list(cex = 0.95)))

glmnetModel <- glmnet(as.matrix(training[, fullSet]),
                      y = training$Class, family = 'binomial')

predict(glmnetModel, 
        newx = as.matrix(training[1:5, fullSet]),
        s = c(0.05, 0.1, 0.2),
        type = 'class')
# tell me which predictors were used in the model.
predict(glmnetModel, 
        newx = as.matrix(training[1:5, fullSet]),
        s = c(0.05, 0.1, 0.2),
        type = 'nonzero')

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 40))
set.seed(476)
glmnTuned <- train(training[, fullSet],
                   training$Class,
                   method = 'glmnet', 
                   tuneGrid = glmnGrid,
                   preProc = c('center', 'scale'),
                   metric = 'ROC', trControl = ctrl)

plot(glmnTuned, plotType = 'level')

sparseLdaModel <- sda(as.matrix(training[, fullSet]),
                      training$Class, 
                      lambda = 0.01, 
                      stop = -6)
# or ... takes more than 30 minutes.
sldaTuned <- train(as.matrix(training[, fullSet]),
                   y = training$Class,
                   method = 'sparseLDA',
                   tuneLength = 20, 
                   trControl = ctrl)
sldaTuned
#### Nearest Shrunken centroids
# these knuckleheads are requiring us to transpose X

nscModel <- pamr.train(data = list(
                                   x = t(training[, fullSet]),
                                   y = training$Class
                                   ))
# extract predictions
pamr.predict(nscModel, newx = t(training[1:5, fullSet]), threshold = 5)
# tell me which variables were used for a given threshold
thresh17Vars <- pamr.predict(nscModel, newx = t(training[1:5, fullSet]),
                             threshold = 17, type = 'nonzero')

# using caret
nscGrid <- data.frame(.threshold = 0:25)
set.seed(476)
nscTuned <- train(training[, fullSet], training$Class,
                  method = 'pam', 
                  preProc = c('center', 'scale'),
                  tuneGrid = nscGrid, 
                  metric = 'ROC',
                  trControl = ctrl)

# Show variables used at chosen threshold
predictors(nscTuned)
varImp(nscTuned, scale = FALSE)

