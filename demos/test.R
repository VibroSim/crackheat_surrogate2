library(DiceKriging)

fundet <- function(x)  return((sin(10 * x)/(1 + x) + 2*cos(5 * x) * x^3 + 0.841)/1.6)
theta <-  .2 #1/sqrt(30)
n <- 7
x <- seq(0, 1, length = n)
t <- seq(0, 1, by = 0.01)
t <- sort(c(t, x))
repart <- c(150, 30, 70, 100, 10, 300, 40)
noise.var <- 4/repart
z <- fundet(x)
y <- z + sqrt(noise.var) * rnorm(length(x))
#model <- km(design = data.frame(x = x), response = y, coef.trend = 0, coef.cov = theta, coef.var = 0.03, noise.var = noise.var)
#model <- km(formula= ~x+I(x^2),design = data.frame(x = x), response = y, coef.cov = theta, coef.var = 0.03, noise.var = noise.var)
#model <- km(formula= ~x+I(x^2)+I(x^3),design = data.frame(x = x), response = y, coef.cov = theta, coef.var = 1.0, noise.var = noise.var)
#model <- km(formula= ~1,design = data.frame(x = x), response = y, coef.cov = theta, coef.var = 0.03, noise.var = noise.var)
model <- km(formula= ~x+I(x^2),design = data.frame(x = x), response = y, coef.cov=theta, coef.var=.03, noise.var = noise.var)

p <- predict.km(model, newdata = data.frame(x = t), type = "UK")

plot(t,p$mean,ylim=c(-1.0,1.8))
points(x,y)
lines(t,fundet(t))
lines(t,p$lower95)
lines(t,p$upper95)
