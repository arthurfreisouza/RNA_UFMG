rm(list=ls())

ydelay <- function(x,nx)
  
{
  
  # x is a Nx1 matrix, where N is the number of samples
  
  N<-length(x)
  
  xmat<-matrix(0,nx+1,N+nx) # Zeros matrix
  
  xmat[1,(1:N)]<-x
  
  for (i in (1:nx))
    
  {
    
    ci<-i
    
    cf<-ci+N-1
    
    xmat[(i+1),(ci+1):(cf+1)]<-xmat[i,(ci:cf)]
    
  }
  
  return(xmat[,(1:N)])
  
}

data("AirPassengers")
ccf(AirPassengers, AirPassengers)
Air_13<-ydelay(AirPassengers,13)

#teste<- rbind(Air_11,Air_12,Air_13)
plot(decompose(AirPassengers))

library(RSNNS)

entrada<-as.matrix(Air_13[12:14,14:ncol(Air_13)])
entrada<-t(entrada)
saida<-as.matrix(Air_13[1,14:ncol(Air_13)])


dim(entrada)
dim(saida)

dados<-cbind(entrada,saida)

dados<-dados[sample(nrow(dados)),]

dados<- splitForTrainingAndTest(dados[,1:3], dados[,4], ratio=0.5)
dados<- normTrainingAndTestSet(dados,dontNormTargets = TRUE)




model <- mlp(dados$inputsTrain, 
             dados$targetsTrain, 
             size=50, 
             learnFuncParams = 0.001, 
             maxit=100, 
             inputsTest = dados$inputsTest, 
             targetsTest = dados$targetsTest, 
             linOut=TRUE)

predictions <- round(predict (model, dados$inputsTest))
#print(cbind(dados$targetsTest, predictions))
#par(new = TRUE)




x_lim <- c(1, 131)
y_lim <- range(c(dados$targetsTrain, dados$targetsTest))
plot(seq(from = 1, to = 65, by = 1), 
     dados$targetsTrain, 
     type = 'l', 
     col = 'black', 
     xlim = x_lim, 
     ylim = y_lim, 
     xlab = "X", 
     ylab = "Y", 
     main = "Plot Contínuo")

par(new = TRUE)

plot(seq(from = 66, to = 131, by = 1), 
     dados$targetsTest, 
     type = 'l', 
     col = 'blue', 
     xlim = x_lim, 
     ylim = y_lim, 
     xlab = "", 
     ylab = "", 
     axes = FALSE)  # Desativar eixos para o segundo gráfico
          
par(new = TRUE)

plot(seq(from = 66, to = 131, by = 1), 
     predictions, 
     type = 'l', 
     col = 'red', 
     xlim = x_lim, 
     ylim = y_lim, 
     xlab = "", 
     ylab = "", 
     axes = FALSE)  # Desativar eixos para o segundo gráfico
