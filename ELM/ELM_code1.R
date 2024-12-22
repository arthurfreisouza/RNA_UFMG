  rm(list = ls())
  library(corpcor)
  
  # Geração dos dados que irão ser usados para treinar.
  N <- 30
  m1 <- c(2, 2)
  m2 <- c(4, 4)
  m3 <- c(2, 4)
  m4 <- c(4, 2)
  # Matriz com 2 colunas, cada uma com 30 valores cada, gerada randomicamente e acrescidos dos valores (2, 2) e (4, 4).
  # Isso serve para deslocar o centro dos pontos.
  g1 <- matrix(rnorm(N*2, sd = 0.6), nrow = N, ncol = 2) + matrix(m1, nrow = N,  ncol = 2, byrow = TRUE )
  g2 <- matrix(rnorm(N*2, sd = 0.6), nrow = N, ncol = 2) + matrix(m2, nrow = N,  ncol = 2, byrow = TRUE )
  
  # Matriz com 2 colunas, cada uma com 30 valores cada, gerada randomicamente e acrescidos dos valores (2, 4) e (4, 2).
  # Isso serve para deslocar o centro dos pontos.
  g3 <- matrix(rnorm(N*2, sd = 0.6), nrow = N, ncol = 2) + matrix(m3, nrow = N,  ncol = 2, byrow = TRUE )
  g4 <- matrix(rnorm(N*2, sd = 0.6), nrow = N, ncol = 2) + matrix(m4, nrow = N,  ncol = 2, byrow = TRUE )
  
  xc1 <- rbind(g1, g2)
  xc2 <- rbind(g3, g4)
  plot(xc1[, 1], xc1[, 2], col = 'red', xlim = c(0, 6), ylim = c(0, 6))
  par(new = TRUE)
  plot(xc2[, 1], xc2[, 2], col = 'blue', xlim = c(0, 6), ylim = c(0, 6))

  X <- rbind(xc1, xc2)
  y_onesneg <- matrix(-1, ncol = 1, nrow = 2*N)
  y_onespos <- matrix(1, ncol = 1, nrow = 2*N)
  Y <- rbind(y_onesneg, y_onespos)
  
  
  # Terei 100 neuronios na camada intermediária.
  # Será 2 valores de entradas e 1 bias.
  p <- 1
  Z <- matrix(runif(3*p, -0.5, 0.5), nrow = 3, ncol = p)
  
  # Realizando o produto matricial e obtendo H com a função de ativação.
  Xaug <- cbind(X, 1)
  H <- tanh(Xaug %*% Z)
  
  # Agora, podemos utilizar w = H⁺Y com a solução da pseudoinversa.
  Haug <- cbind(H, 1)
  pseudHaug <- pseudoinverse(Haug)
  w <- pseudHaug %*% Y
  
  Y_hat_train <- sign(Haug %*% w)
  e_train <- sum((Y - Y_hat_train)^2) / 4
  

  seqx1x2 <- seq(-2, 10, 0.1) # gera valores de 0 até 6 com passo de 0.2.
  npgrid <- length(seqx1x2) # Pegando o tamanho da sequencia de valores gerados.
  MZ <- matrix(nrow = npgrid, ncol = npgrid) # Criando 1 matriz quadrada.
  for (i in 1:npgrid){
    for (j in 1:npgrid){
      x1 <- seqx1x2[i]
      x2 <- seqx1x2[j]
      x1x2 <- as.matrix(cbind(x1, x2, 1))
      h1 <- cbind(tanh(x1x2 %*% Z), 1)
      MZ[i, j] <- sign(h1 %*% w)
    }
  }
  
contour(seqx1x2, seqx1x2, MZ, lebels = 1, xlim = c(0, 6), ylim = c(0, 6), levels = 1)
par(new = TRUE)
plot(xc1[, 1], xc1[, 2], col = 'red', xlim = c(0, 6), ylim = c(0, 6))
par(new = TRUE)
plot(xc2[, 1], xc2[, 2], col = 'blue', xlim = c(0, 6), ylim = c(0, 6))


err_list <- vector()

for (aux in 1:length(Y_hat_train)){
  err_list[aux] <- (Y[aux] - Y_hat_train[aux])
}

err_list <- as.array(err_list)
err_list
