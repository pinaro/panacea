library(dplyr)
library(readr)
library(scales)
library(ggplot2)

path <- "/Users/pinar/git/Safe-Secure-RL/grid-world/experiment/"

dat <- read_delim(paste0(path,"fabricated-trajectories.csv"),
                  delim = '\t', 
                  col_types = cols(.default = col_double(), CI = col_character(), Weighting = col_character()))
dat$Estimator = paste(dat$CI, dat$Weighting, sep=", ")
J_pi_b <-unique(dat$J_pi_b)
J_pi_e <- unique(dat$J_pi_e)

# Panacea data
panacea <- read_delim(paste0(path,"panacea.csv"),
                  delim = '\t', 
                  col_types = cols(.default = col_double(), CI = col_character(), Weighting = col_character()))
panacea$Estimator = paste(panacea$CI, panacea$Weighting, sep=", ")
panacea$Estimator = paste(panacea$Estimator, ', Panacea', sep="")

dat <- cbind.data.frame(k=dat$K, Estimator=dat$Estimator, J_hat_pi_e=dat$J_hat_pi_e)
panacea <- cbind.data.frame(k=panacea$k, Estimator=panacea$Estimator, J_hat_pi_e=panacea$J_hat_pi_e)
dat <- rbind.data.frame(dat, panacea)

pdf(paste0(path,"../../Security-Analysis-of-Safe-and-Seld-RL-Alg/figures/grid-experiment-change-k-c.pdf"),w=5,h=4)
ggplot(data=dat)+theme_gray(14)+ # %>% filter(dat$K <= 200)
  aes(x=k, y=J_hat_pi_e, group=factor(Estimator), color=factor(Estimator)) +
  #geom_point(size=0.1)+
  geom_line(size=1)+
  geom_jitter(size=0.7)+ 
  xlab("k")+
  ylab(expression(paste(L, "(", pi[e],", ", D[s], ")")))+#"*"^",", 
  #geom_errorbar(aes(ymin=1-avg+ci,ymax=1-avg-ci), width=0.025, alpha=0.5)+
  scale_x_continuous(breaks=pretty_breaks(10))+
  scale_y_continuous(breaks=pretty_breaks(8))+ #math_format(10^.x)labels=scientific_format(digits=1)
  #geom_hline(aes(yintercept=J_pi_b), linetype=2, color='black')+#size = 2, 
  #annotate('text', x = 200, y = J_pi_b,
  #         color = 'black', size = 3, parse = TRUE, vjust = -0.5,
  #         label = expression(paste("J(", pi[b], ")")))+
  geom_hline(aes(yintercept=J_pi_e), linetype=2, color='black')+
  annotate('text', x = 8000, y = 0.35,
           color = 'black', size = 5, parse = TRUE, vjust = -0.5,
           label = expression(paste("J(", pi[e], ")")))+
  theme(legend.position="top",
        legend.box.margin = margin(-10,-10,-10,-10),
        legend.title = element_blank())+
  guides(col = guide_legend(nrow=3))+
  coord_cartesian(ylim = c(-0.2, 1.2))+
  scale_color_discrete(breaks=c("CH, IS, Panacea", "CH, WIS, Panacea", "AM, IS, Panacea", "CH, IS", "CH, WIS", "AM, IS", "AM, WIS"))
  #expand_limits(x = 0, y = 0)
  #scale_y_continuous(trans = log2_trans(),
  #                 breaks = trans_breaks("log2", function(x) 2^x),
  #                 labels = trans_format("log2", math_format(2^.x)))
  #scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
  #              labels = scales::trans_format("log10", scales::math_format(10^.x)))
  #scale_y_log10(breaks=log_breaks(),
  #              label=percent_format(accuracy = .001),
  #              limits=c(0.0001,1))
  #facet_wrap(blkSize ~ bound, ncol=3,scales='free')
  #facet_grid(blkSize ~ ., scales='free')
#geom_hline(yintercept=.999, linetype=2)
#scale_y_continuous(lim=c(.99,NA))
dev.off()

###################################################################################
library(dplyr)
library(readr)
library(scales)
library(ggplot2)

path <- "/Users/pinar/git/Safe-Secure-RL/grid-world/experiment/"

dat <- read_delim(paste0(path,"fabricated-trajectories-panacea.csv"),
                  delim = '\t', 
                  col_types = cols(.default = col_double(), CI = col_character(), Weighting = col_character()))
dat$Estimator = paste(dat$CI, dat$Weighting, sep=",")
J_pi_b <-unique(dat$J_pi_b)
J_pi_e <- unique(dat$J_pi_e)
k <- 8200

pdf(paste0(path,"../../Security-Analysis-of-Safe-and-Seld-RL-Alg/figures/grid-experiment-change-c.pdf"),w=5,h=4)
ggplot(data=dat)+theme_gray(14)+ # %>% filter(dat$K <= 200)
  aes(x=C, y=J_hat_pi_e, group=Estimator, color=Estimator)+
  geom_point(size=0.5)+
  geom_line()+
  xlab("c'")+
  ylab(expression(paste(hat(J), "(", pi[e], ")")))+#"*"^",", 
  #geom_errorbar(aes(ymin=1-avg+ci,ymax=1-avg-ci), width=0.025, alpha=0.5)+
  scale_x_continuous(breaks=pretty_breaks(10))+
  scale_y_continuous(breaks=pretty_breaks(8))+
  geom_hline(aes(yintercept=J_pi_e), linetype=2, color='black')+
  annotate('text', x = 50, y = 0.45,
           color = 'black', size = 5, parse = TRUE, vjust = -0.5,
           label = expression(paste("J(", pi[e], ")")))+
  theme(legend.position="top",
        legend.box.margin = margin(-10,-10,-10,-10),
        legend.title = element_blank())+
  coord_cartesian(xlim = c(0, 20), ylim = c(-0.3, 2))
dev.off()
