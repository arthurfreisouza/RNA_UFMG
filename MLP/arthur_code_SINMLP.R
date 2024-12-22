rm(list=ls())

sech2<- function (u)
{
  return (((2/(exp(u)+exp(-u)))*(2/(exp(u)+exp(-u)))))
  
}

x <- as.matrix(seq(0, 2*pi, 0.1*pi)) 
y <- sin(x)
plot(x, y, type = 'l')

w14 <- runif(1)-0.5 
w15 <- runif(1)-0.5 
w16 <- runif(1)-0.5 

w24 <- runif(1)-0.5 
w25 <- runif(1)-0.5 
w26 <- runif(1)-0.5 

w37 <- runif(1)-0.5 
w47 <- runif(1)-0.5 
w57 <- runif(1)-0.5 
w67 <- runif(1)-0.5 


i1 <- 1
i3 <- 1

nepocas<-0
tol<-0.001
eepoca<-tol+1
maxepocas<-100000
N<-nrow(x)
eta<-0.001

evec<-matrix(nrow=1, ncol=maxepocas)

while((nepocas<maxepocas) &&(eepoca>tol))
{
  
  ei2<-0
  xseq<-sample(N)
  for (i in 1:N){
    
    irand<-xseq[i]
    i2 <- x[irand,1]
    
    y7 <- y[irand, 1]
    
    u4 <- (i1*w14 + i2*w24)
    i4 <- tanh(u4)
    
    
    u5 <- (i1*w15 + i2*w25)
    i5 <- tanh(u5)
    
    
    u6 <- (i1*w16 + i2*w26)
    i6 <- tanh(u6)
    
    u7 <- i3*w37 + i4*w47 + i5*w57 + i6*w67
    i7 <-u7
    
    e7 <- y7 - i7
    d7 <- e7*1
    
    dw37 <- eta*d7*i3
    dw47 <- eta*d7*i4
    dw57 <- eta*d7*i5
    dw67 <- eta*d7*i6
    
    d4 <- d7*w47*sech2(u4)
    d5<- d7*w47*sech2(u5)
    d6<- d7*w47*sech2(u6)
    
    
    
    
    dw14 <- eta*d4*i1
    dw15 <- eta*d5*i1
    dw16 <- eta*d6*i1
    
    
    dw24<- eta*d4*i2
    dw25 <- eta*d5*i2
    dw26 <- eta*d6*i2
    
    
    
    
    w14 <- w14 + dw14
    w15 <- w15 + dw15 
    w16 <- w16 + dw16 
    
    w24 <- w24 + dw24 
    w25 <- w25+ dw25
    w26 <- w26+ dw26
    
    w37 <- w37 + dw37
    w47 <-  w47 + dw47 
    w57 <-  w57 + dw57 
    w67 <-  w67 + dw67
    
    
    ei2<-ei2+((e7^2))/N
    
  }
  
  nepocas<-nepocas+1
  evec[nepocas]<-ei2/N
  
  eepoca<-evec[nepocas]
  
}


plot (seq(1, nepocas, 1), evec[1:nepocas], type = 'l')






xrange <- seq(0, 2*pi, 0.01*pi)
yhat  <- matrix(nrow = length(xrange), ncol = 1)
N2<- length(xrange)

for (i in 1:N2)
{
  i2 <- xrange[i]
  
  
  u4 <- (i1*w14 + i2*w24)
  i4 <- tanh(u4)
  
  
  u5 <- (i1*w15 + i2*w25)
  i5 <- tanh(u5)
  
  
  u6 <- (i1*w16 + i2*w26)
  i6 <- tanh(u6)
  
  u7 <- i3*w37 + i4*w47 + i5*w57 + i6*w67
  i7 <-u7
  
  yhat[i] <- i7
  
}

plot(xrange,yhat,type='l',xlim = c(0,2*pi) ,col = 'red',ylim = c(-1,1))
par(new = T)
plot(xrange,sin(xrange),type='l',xlim = c(0,2*pi),col = 'blue',ylim = c(-1,1))
par(new = T)
plot(x,y,type='l',xlim = c(0,2*pi) ,col = 'black',ylim = c(-1,1))


  