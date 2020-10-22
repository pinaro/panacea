library(dplyr)
library(readr)
library(scales)
library(ggplot2)

DELTA = 0.05
dat <- data.frame("N" = 1:1000)
dat$Val <- (log(2/DELTA)) * (dat$N+1)
dat$Val <- 1 - sqrt(dat$Val / 2)
G_min <- 0
G_max <- 1

ggplot(data=dat%>%filter(dat$Val >= 0))+theme_gray(16)+ 
  aes(x=N, y=Val)+
  geom_point(size=0.25)+
  geom_line()+
  xlab("Num of data points")+
  ylab("")+
  scale_x_continuous()+
  scale_y_continuous(breaks=pretty_breaks(5))+
  geom_hline(aes(yintercept=G_min), linetype=2, color='black')+
  geom_hline(aes(yintercept=G_max), linetype=2, color='black')

###################################################################################
DELTA = 0
dat <- data.frame("N" = 2:1000000)
dat$Val <- log(2/DELTA) / (2*dat$N)
#dat$Val <- sqrt(dat$Val)
dat$Val <- (dat$N-1/(dat$N)) + sqrt(dat$Val)
sum(dat$Val > 1) == nrow(dat)

ggplot(data=dat)+theme_gray(16)+ 
  aes(x=N, y=Val)+
  geom_point(size=0.25)+
  geom_line()+
  xlab("Num of data points in dataset")+
  ylab("I")+
  scale_x_continuous()+
  scale_y_continuous(breaks=pretty_breaks(5))+
  #geom_hline(aes(yintercept=G_min), linetype=2, color='black')+
  #geom_hline(aes(yintercept=G_max), linetype=2, color='black')

###################################################################################
N = 100
dat <- data.frame("K" = 2:50000)
G <- runif(N)
I <- runif(N, min=0, max=1e+100)
point <- sum(I*G) / sum(I)
dat$bool <- (dat$K-1 / dat$K) > point

ggplot(data=dat)+theme_gray(16)+ 
  aes(x=N, y=Val)+
  geom_point(size=0.25)+
  geom_line()+
  xlab("Num of data points")+
  ylab("")+
  scale_x_continuous()+
  scale_y_continuous(breaks=pretty_breaks(5))+
  geom_hline(aes(yintercept=G_min), linetype=2, color='black')+
  geom_hline(aes(yintercept=G_max), linetype=2, color='black')

###################################################################################
DELTA = 0.5
N = 100000
TMP <- (log(2/DELTA)) / (2*N)

dat1 <- data.frame("first" = 1:(N-1))
dat1$first <- (dat1$first/N) + sqrt(TMP)
dat1$val <- 1
dat <- data.frame("first" = apply(dat1, 1, FUN=min))

dat2 <- data.frame("second" = 0:(N-2))
dat2$second <- (dat2$second/N) + sqrt(TMP)
dat2$val <- 1
dat$second <- apply(dat2, 1, FUN=min)

dat$result <- dat1$first - dat$second
dat$I <- 1:(N-1)
sum(dat$result > 0) == nrow(dat)

ggplot(data=dat)+# %>% filter(dat$I < 10000))+theme_gray(16)+ #
  aes(x=I, y=result)+
  geom_point(size=0.25)+
  geom_line()+
  xlab("Num of data points")+
  ylab("")+
  scale_x_continuous()+
  scale_y_continuous(breaks=pretty_breaks(5))+
  geom_hline(aes(yintercept=1/N), linetype=2, color='red')

###################################################################################
DELTA = 1
N = 100

dat1 <- data.frame("first" = 0:N)
tmp <- (log(2/DELTA)) / (2*N)
dat1$first <- (dat1$first/N) + sqrt(tmp)
dat1$val <- 1
temp1 <- data.frame("first" = apply(dat1, 1, FUN=min), "side"='LHS', N=0:N)

dat2 <- data.frame("second" = 0:N)
tmp <- (log(2/DELTA)) / (2*(N+1))
dat2$second <- (dat2$second+1/N+1) + sqrt(tmp)
dat2$val <- 1
temp2 <- data.frame("first" = apply(dat2, 1, FUN=min), "side"='RHS', N=0:N)

dat <- rbind(temp1, temp2)

#pdf(paste0(path,"/Users/pinar/Desktop/trial.pdf"),w=5,h=4)
ggplot(data=dat)+theme_gray(16)+ #%>%filter(dat$N>9000)
  aes(x=N, y=first, color=side, group=side)+
  geom_point(size=0.25)+
  geom_line()+
  #xlab("Num of data points")+
  #ylab("")+
  #scale_x_continuous()+
  #scale_y_continuous(breaks=pretty_breaks(5))
#dev.off()

