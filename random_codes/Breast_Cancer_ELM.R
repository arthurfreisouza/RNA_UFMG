rm(list = ls())
source("/home/arthur/Desktop/RNA_BRAGA/ELM/ELM_FUNCTIONS.R") # Carregando o arquivo das funções.
library(mlbench)
library(Rcpp)
library(RSNNS)
data("BreastCancer")
xyall <- na.omit(data.matrix(BreastCancer)) # Emitindo os NaN's, há várias formas de tratar isso.
xy <- splitForTrainingAndTest(xyall[, (2 : 10)], 2*(xyall[, 11] - 1.5), ratio = 0.3) # Passando o label para valores (-1, 1)

X <- xy$inputsTrain
Y <- xy$targetsTrain
X_test <- xy$inputsTest
Y_test <- xy$targetsTest

p <- 50
retELM <- trainELM(X, Y, p, 1)
w <- retELM[[1]]
z <- retELM[[3]]


y_hat_test <- YELM(X_test, z, w, 1)
print(sum((Y_test - y_hat_test)^2) / 4)


MC <- table(y_hat_test, Y_test)
print(MC)