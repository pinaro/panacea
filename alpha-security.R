library(dplyr)
library(readr)
library(scales)
library(ggplot2)

path <- "/Users/pinar/git/Safe-Secure-RL/grid-world/experiment/"

dat <- read_delim(paste0(path,"max_alpha.csv"),
                  delim = '\t')
dat$Estimator = paste(dat$CI, dat$Weighting, sep=", ")

pdf(paste0(path,"../../Security-Analysis-of-Safe-and-Seld-RL-Alg/figures/max_alpha_security.pdf"),w=5,h=4)
ggplot(data=dat)+ #
  theme_gray(14)+
  theme(text = element_text(size=12), axis.text.x = element_text(angle=45, hjust=1))+
  aes(x=K, y=Result, group=Estimator, color=Estimator)+
  geom_point(size=0.1)+
  geom_line()+
  geom_jitter(size=0.2)+ 
  xlab("k")+
  ylab(expression(alpha^"*"))+#alpha[paste(n, ",", k)]^"*"
  facet_wrap(N~., scales = 'free', labeller = label_both)+
  scale_y_continuous(breaks=pretty_breaks(8))+
  scale_x_continuous(breaks=pretty_breaks(5))+
  #scale_color_discrete(name = 'Estimator')+
  theme(legend.position="top",
        legend.box.margin = margin(-10,-10,-10,-10),
        legend.title = element_blank())
dev.off()

###################################################################################
pdf(paste0(path,"../../Security-Analysis-of-Safe-and-Seld-RL-Alg/figures/alpha_security.pdf"),w=5,h=4)
ggplot(data=dat)+ #
  theme_gray(14)+ 
  aes(x=K, y=Result, group=factor(N), color=factor(N))+
  geom_point(size=0.1)+
  geom_line()+
  xlab("k")+
  ylab(expression(alpha[paste(n, ",", k)]^"*"))+
  facet_grid(CI ~ Weighting, scales = 'free')+
  scale_color_discrete(name = 'n')+
  theme(legend.position="top",
        legend.box.margin = margin(-10,-10,-10,-10))
dev.off()