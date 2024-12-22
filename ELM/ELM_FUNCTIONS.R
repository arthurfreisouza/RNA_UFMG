rm(list = ls())

trainELM <- function(xin, yin, p, control){
  library(corpcor)
  n <- dim(xin)[2] # Pegando o número de colunas da entrada(número de valores de entrada da rede.)
  if (control == 1){
    xin <- cbind(xin, 1)
    Z <- matrix(runif((n + 1) * p, -0.5, 0.5), nrow = (n + 1), ncol = p)
  }
  else{
    Z <- matrix(runif((n * p), -0.5, 0.5), nrow = n, ncol = p)
  }
  H <- tanh(xin %*% Z)
  Hout <- cbind(H, 1) # As entradas da camada de saída.
  W <- pseudoinverse(Hout) %*% yin
  return(list(W, H, Z))
}

# Criando a função que fará a saída de uma rede neural ELM : 

YELM <-  function(xin, Z, W, control){
  if (control == 1){
    xin = cbind(xin, 1)
  }
  H <- tanh(xin %*% Z)  # Fazendo a projeção na camada intermediária.
  Hout <- cbind(H, 1)
  Y_hat <- sign(Hout %*% W)
  
  return (Y_hat)
}