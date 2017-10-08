library(matlib)

numDataPts <- 10
numDepVars <- 2
betaOpt <- c(1,2)
eps <- .01
p <- 1

Y <- matrix(, nrow = numDataPts, ncol = 1)
X <- matrix(, nrow = numDataPts, ncol = numDepVars)

for(row in 1:numDataPts){
	X[row,] <- (sample.int(101,size=numDepVars,replace=TRUE)-1)/100
	Y[row,1] <- ((betaOpt %*% X[row,])+rnorm(1,0,.2))
}

betaGuess <- matrix(c(6,5),nrow=numDepVars,ncol=1)
W <- diag(1,numDataPts,numDataPts)

error <- 10000
errorOld <- -10000
unchanged <- FALSE
iterations <- 1

while (error > eps & !unchanged & iterations<10){
	print(betaGuess)
	betaGuess <- solve(t(X) %*% W %*% X ) %*% t(X) %*% W %*% Y
	resids <- as.vector(Y - X %*% betaGuess)
	weightVector <- sapply(resids, function(x) (max(.1,abs(x)))^(p-2))
	W <- diag(weightVector)
	error <- weightVector %*% (resids*resids)
	error <- error[1]
	unchanged <- (abs(errorOld-error)<eps)
	errorOld <- error
	iterations <- iterations + 1
}

print(betaGuess)
print(error)