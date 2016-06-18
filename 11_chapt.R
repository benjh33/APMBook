### Classification

library(ggplot2)
library(dplyr)
library(gridExtra)
library(caret)
library(randomForest)
library(AppliedPredictiveModeling)
library(MASS)
library(pROC)

PREVALENCE <- 0.01

# Sensitivity
# Predict true and is true
SENS <- 0.9

# Specificity
# Predict false and is false
SPEC <- 0.98

#' Positive Predicted Value
#' Probability true given a positive test result
#' as specificity gets higher, the predictive
#' power of a positive result approaches 1, regardless
#' of how rare.
ppv <- function(sens, spec, prev) {
  (sens * prev)/
    ((sens * prev) + (1 - spec)*(1 - prev))
}

#' Negative Predicted Value
#' Probability false given negative test result
#' As sensitivity gets higher, probability of false
#' given false test approaches 1.
npv <- function(sens, spec, prev) {
  (spec * (1 - prev))/
    ((prev * (1 - sens)) + (spec * (1 - prev)))
}

m <- expand.grid(sens = seq(0.6, 0.99, length = 6),
                 spec = seq(0.6, 0.99, length =6),
                 prev = seq(0.01, 0.99, length = 50),
                 npv = NA,
                 ppv = NA)

m[, 4:5] <- sapply(list(npv, ppv), function(f) {
  apply(as.matrix(m[,1:3]), 1, function(x) f(x[1], x[2], x[3]))
})

m <- reshape2::melt(m, id.vars = c('sens', 'spec', 'prev'))

m <- m[order(m$prev), ]

print_labels <- function(df, f = label_both) {
  print(df)
  f(df)
}


ggplot(m, aes(x = prev, y = value, color = variable)) +
  geom_line() + theme_bw() + 
  facet_grid(sens ~ spec, labeller = print_labels)

### Probability cost function
#' caret::probFunction
#' cost of futile effort /
#' sum(cost missed opportunity, cost futile effort)
#' proportion of total cost associated with a false
#' positive sample.
#' It is the cost of applying treatment to get a positive
#' result multiplied by the prior of a positive result,
#' divided by the prior times the cost of missing a conversion
#' plus the complement of the prior times the cost of 
#' applying treatment. The denominator is "total cost"
#' in an economic sense - actual spend plus missed opportunity.
#' the numerator is the cost of 
#' @param cost_fn the cost of predicting an event as a non-event - this is often high - ie. acquiring customers.
#' (missing out on a conversion for lack of trying)
#' @param cost_fp cost of a false positive (sending a mailer
#' to a disinterested person) - low nowadays.
pcf <-function(prior, cost_fn, cost_fp) {
  (prior * cost_fp) / 
    # (how much spent on futile effort * prior)
    (prior * cost_fn + (1 - prior) * cost_fp)
    # (missed potential revenue * prior) +
    # (expected cost of trying to get non-converters)
}

pcf_df <- expand.grid(prior = seq(0.01, 1, length = 16),
                      cost_fn = seq(1, 50, length = 50),
                      cost_fp = seq(1, 50, length = 50))
pcf_df$pcf <- apply(pcf_df, 1, function(x) {
  pcf(x[1], x[2], x[3])
})

heatmap_b <- function(df) {
  ggplot(df, aes(x = cost_fn, y = cost_fp)) + 
    geom_tile(aes(fill = pcf)) + 
    scale_fill_gradient(name = 'pcf', 
                        low = 'white', high = 'green') + 
    theme(axis.title.y = element_blank()) + 
    facet_wrap(~prior, ncol = 4) +
    theme_bw()
}
hmaps <- pcf_df %>% group_by(prior) %>%
  do(plot = heatmap_b(.))

hmap_grob <- gridExtra::arrangeGrob(grobs = hmaps[[2]], 
                                    ncol = 4)
plot(hmap_grob)


## Normalized Expected Cost
#' accounts for the prevalence of events, model performance
#' and costs then scales between 0 and 1.
#' PCF (now understood) is multiplied by the probability
#' of a true positive and summed with 1 - PCF times the 
#' probability of a false positive.
nec <- function(pcf, true_p, false_p) {
  pcf * (1 - true_p) + (1 - pcf) * false_p
}

confused <- matrix(sapply(1:1000, function(x) {
  x <- runif(4)
  x/sum(x)
}), byrow = TRUE, nrow = 1000)

colnames(confused) <- c('TP', 'FP', 'FN', 'TN')



## Computig page 266

set.seed(975)
simulatedTrain <- quadBoundaryFunc(500)
simulatedTest <- quadBoundaryFunc(1000)
head(simulatedTrain)

rfModel <- randomForest(class ~ X1 + X2, 
                        data = simulatedTrain,
                        ntree = 2000)
qdaModel <- qda(class ~ X1 + X2, data = simulatedTrain)

qdaTrainPred <- predict(qdaModel, simulatedTrain)
names(qdaTrainPred)
head(qdaTrainPred$class)
head(qdaTrainPred$posterior)

qdaTestPred <- predict(qdaModel, simulatedTest)
simulatedTrain$QDAprob <- qdaTrainPred$posterior[, 'Class1']
simulatedTest$QDAprob <- qdaTestPred$posterior[, 'Class1']

rfTestPred <- predict(rfModel, simulatedTest, type = 'prob')
rfTrainPred <- predict(rfModel, simulatedTrain, type = 'prob')
simulatedTest$RFprob <- rfTestPred[, "Class1"]
simulatedTest$RFclass <- predict(rfModel, simulatedTest)
simulatedTrain$RFprob <- rfTrainPred[, "Class1"]
simulatedTrain$RFclass <- predict(rfModel, simulatedTrain)

## sensitivity and specificity
rfsense <- sensitivity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            positive = 'Class1')

rfspec <- specificity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            negative = 'Class2')

rftable <- prop.table(table(rf = simulatedTest$RFclass, 
                 class = simulatedTest$class))
abs(rftable[1, 1] / sum(rftable[,1]) - rfsense) < 1e-10
abs(rftable[2, 2] / sum(rftable[,2]) - rfspec) < 1e-10


rfpospred <- posPredValue(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            positive = 'Class1')

rfnegpred <- negPredValue(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            negative = 'Class2')

prevalence <- prop.table(table(simulatedTest$class))

abs(rfsense * prevalence['Class1'] / 
  (rfsense * prevalence['Class1'] +
     (1 - rfspec) * prevalence['Class2']) - rfpospred) < 1e-10

abs(rfspec * prevalence['Class2'] /
    ((1- rfsense) * prevalence['Class1'] +
      rfspec * prevalence['Class2']) - rfnegpred) < 1e-10

confusionMatrix(data = simulatedTest$RFclass,
                reference = simulatedTest$class, 
                positive = 'Class1')

rocCurve <- roc(response = simulatedTest$class,
                predictor = simulatedTest$RFprob,
                levels = rev(levels(simulatedTest$class)))

auc(rocCurve)
ci(rocCurve)

plot(rocCurve, legacy.axes = TRUE)

# lift charts

labs <- c(RFprob = "Random Forest", 
          QDAprob = "Quadratic discriminant analysis")
liftCurveGain <- lift(class ~ RFprob + QDAprob, 
                  data = simulatedTest,
                  plot = 'gain',
                  labels = labs)
liftCurveLift <- lift(class ~ RFprob + QDAprob, 
                   data = simulatedTest,
                   labels = labs)

xyplot(liftCurveLift,
       auto.key = list(columns = 2,
                       lines = TRUE, 
                       points = FALSE))

plot(liftCurveGain)
plot(liftCurveLift)
help(lift)

# Calibration

calCurve <- calibration(class ~ RFprob + QDAprob,
                        data = simulatedTest)
calCurve
xyplot(calCurve,
       auto.key = list(columns = 2,
                       lines = TRUE, 
                       points = FALSE))
sigmoidalCal <- glm(relevel(class, ref = "Class2") ~ QDAprob,
                    data = simulatedTrain,
                    family = binomial)
coef(summary(sigmoidalCal))
simulatedTest$QDAsigmoid <- predict(sigmoidalCal,
        newdata = simulatedTest[, 'QDAprob', drop = FALSE],
        type = 'response')
sigmoidalCal <- glm(relevel(class, ref = "Class2") ~ RFprob,
                    data = simulatedTrain,
                    family = binomial)
# algorithm didn't converge because it's already pretty
# well calibrated.
ggplot(simulatedTrain, aes(y = RFprob, x = class)) +
  geom_boxplot()
simulatedTest$RFsigmoid <- predict(sigmoidalCal,
        newdata = simulatedTest[, 'RFprob', drop = FALSE],
        type = 'response')
calibratedPreds <- calibration(class ~ QDAsigmoid + RFsigmoid,
                               data = simulatedTest)
xyplot(calibratedPreds,
       auto.key = list(columns = 2,
                       lines = TRUE, 
                       points = FALSE))

# NaiveBayes
library(klaR)

BayesCal <- NaiveBayes(class ~ QDAprob, data = simulatedTrain,
                       usekernel = TRUE)
BayesProb <- predict(BayesCal, 
     newdata = simulatedTest[, 'QDAprob', drop = FALSE],
     type = 'response')
simulatedTest$QDABayes <- BayesProb$posterior[, 'Class1']

allcalibration <- calibration(class ~ QDAprob + RFprob +
                                QDAsigmoid + QDABayes,
                              data = simulatedTest)
xyplot(allcalibration,
       auto.key = list(columns = 2,
                       lines = TRUE, 
                       points = FALSE))

