library(dplyr)
library(readr)
library(scales)
library(ggplot2)
library(cowplot)
library(grid)
library(gridExtra)

PATH <- "/Users/Pinar/Desktop/NeurIPS_fig1/results/"
WEIGHTING <- 'IS' # pick IS or WIS

dat <- read_delim(paste0(PATH, "panacea_all.csv"),
                  delim = '\t',
                  col_types = cols(.default = col_double(), Estimator = col_character(), Problem = col_character()))

dat$pi_b <- 0
dat$pi_e <- 0
dat$pi_b[dat$Problem == 'Diabetes'] <- 0.21880416377956427
dat$pi_e[dat$Problem == 'Diabetes'] <- 0.14524769684920283
dat$pi_b[dat$Problem == 'Grid-world'] <- 0.7970902655221709
dat$pi_e[dat$Problem == 'Grid-world'] <- 0.7280028364424095

results1 <- dat %>% group_by(k, Estimator, Problem, panaceaAlpha) %>% summarize(avg=mean(Result), var=var(Result), cnt=n(), ci=1.96*sqrt(avg*(1-avg)/n()), pi_e = unique(pi_e), pi_b = unique(pi_b))
# Fix data
results1$Panacea <- 'with\nPanacea'
#results1$Estimator[results1$Estimator == 'IS'] <- 'IS + Panacea'
#results1$Estimator[results1$Estimator == 'WIS'] <- 'WIS + Panacea'

# Read in grid without Panacea
dat <- read_delim(paste0(PATH, "results_all.csv"),
                  delim = '\t', 
                  col_types = cols(.default = col_double(), Estimator = col_character(), Problem = col_character()))

dat$pi_b <- 0
dat$pi_e <- 0
dat$pi_b[dat$Problem == 'Diabetes'] <- 0.21880416377956427
dat$pi_e[dat$Problem == 'Diabetes'] <- 0.14524769684920283
dat$pi_b[dat$Problem == 'Grid-world'] <- 0.7970902655221709
dat$pi_e[dat$Problem == 'Grid-world'] <- 0.7280028364424095

# Fix data
results <- dat %>% group_by(k, Estimator, Problem) %>% summarize(avg=mean(Result), var=var(Result), cnt=n(), ci=1.96*sqrt(avg*(1-avg)/n()), pi_e = unique(pi_e), pi_b = unique(pi_b))
results$Panacea <- 'without\nPanacea'
test <- NULL
for (alpha in unique(results1$panaceaAlpha)) {
  a <- cbind(data.frame(results), data.frame(panaceaAlpha = rep(alpha, nrow(results))))
  test <- rbind(test, a)
}
# Combine data frames and plot
results <- rbind(test, data.frame(results1))

if (WEIGHTING == 'IS') {
  results <- results %>% filter(panaceaAlpha >= 0.1)
  results$panaceaAlpha <- factor(results$panaceaAlpha, levels = c("0.1", "0.5", "1", "5"), labels = c(paste0(expression(alpha), ': 0.1'), paste0(expression(alpha), ': 0.5'), paste0(expression(alpha), ': 1'), paste0(expression(alpha), ': 5')))
} else {
  results <- results %>% filter(panaceaAlpha %in% c(0.01, 0.5, 1))
  results$panaceaAlpha <- factor(results$panaceaAlpha, levels = c("0.01", "0.5", "1"), labels = c(paste0(expression(alpha), ': 0.01'), paste0(expression(alpha), ': 0.5'), paste0(expression(alpha), ': 1')))
}

# Combine data frames and plot
p1 <- ggplot(data=results %>% filter(Estimator == WEIGHTING))+# %>% mutate(group = paste(Problem, WEIGHTING, sep="-")))+
  theme_gray(14)+
  aes(x=k, y=avg, group=factor(Panacea), color=factor(Panacea))+
  geom_point(size=0.02)+
  #geom_jitter(size=0.01)+
  geom_line()+
  #xlab("Number of trajectories added to dataset of size 1,500")+
  xlab("")+
  #ylab("Probability of undesirable policy\n passing safety test")+
  ylab(expression(paste('Mean  ', f^"CH, *")))+
  #geom_errorbar(aes(ymin=avg+var, ymax=avg-var), width=0.025, alpha=0.5)+
  geom_hline(aes(yintercept=pi_b), linetype=2, color='black')+
  #geom_hline(aes(yintercept=pi_b), linetype=2, color='purple')+
  facet_grid(Estimator+Problem~panaceaAlpha, scales = 'free', labeller = label_parsed)+#, labeller = labeller(Estimator = Estimator.labs, panaceaAlpha = panaceaAlpha.labs))+
  #scale_y_continuous(trans = "log")+#breaks=pretty_breaks(4))+#, )+labels=scientific_format(digits=0),
  #scale_y_continuous(trans=weird)+#reverselog_trans(base=10))+
  #scale_y_continuous(breaks=pretty_breaks(5))+
  #scale_x_continuous(breaks=seq(0, 100, by=25))+
  #scale_x_continuous(breaks=pretty_breaks(5))+
  #scale_color_discrete(name = 'Estimator')+
  theme(axis.text.x = element_text(angle=45, hjust=1), 
        axis.text=element_text(size=8),
        legend.box.margin = margin(-10,-10,-10,-10), 
        legend.position = "right")+
  scale_color_discrete(name="")

legend <- get_legend(p1 + theme(legend.box.margin = margin(0, 0, 0, 5)))

p1 <- ggplot(data=results %>% filter(Estimator == WEIGHTING))+# %>% filter(Problem == 'Grid-world'))+# %>% mutate(group = paste(Problem, WEIGHTING, sep="/")))+
  theme_gray(14)+
  aes(x=k, y=avg, group=factor(Panacea), color=factor(Panacea))+
  geom_point(size=0.02)+
  geom_line()+
  #xlab("Number of trajectories added to dataset of size 1,500")+
  xlab("")+
  #ylab("Probability of undesirable policy\n passing safety test")+
  ylab(expression(paste('Mean  ', L^"CH, *")))+
  geom_errorbar(aes(ymin=avg+var, ymax=avg-var), width=0.025, alpha=0.5)+
  geom_hline(aes(yintercept=pi_b), linetype=2, color='black')+
  facet_grid(Estimator+Problem~panaceaAlpha, scales = 'free', labeller = label_parsed)+#, labeller = labeller(Estimator = Estimator.labs, panaceaAlpha = panaceaAlpha.labs))+
  #scale_y_continuous(trans = "log")+#breaks=pretty_breaks(4))+#, )+labels=scientific_format(digits=0),
  #scale_y_continuous(trans=weird)+#reverselog_trans(base=10))+
  scale_y_continuous(breaks=pretty_breaks(6))+
  #scale_x_continuous(breaks=seq(0, 100, by=25))+
  scale_x_continuous(breaks=pretty_breaks(5))+
  #scale_color_discrete(name = 'Estimator')+
  theme(axis.text.x = element_text(angle=45, hjust=1), 
        axis.text=element_text(size=8),
        legend.box.margin = margin(-10,-10,-10,-10), 
        legend.position = "none")+
  scale_color_discrete(name="")

if (WEIGHTING == 'IS') {
  y_min <- -1
  y_max <- 1
} else {
  y_min <- 0.15
  y_max <-0.9
}

p2 <- ggplot(data=results %>% filter(Estimator == WEIGHTING) %>% filter(Panacea == 'with\nPanacea'))+# %>% filter(Problem == 'Grid-world'))+# %>% mutate(group = paste(Problem, WEIGHTING, sep="-")))+
  theme_gray(14)+
  aes(x=k, y=avg, group=factor(Panacea), color=factor(Panacea))+
  geom_point(size=0.02)+
  #geom_jitter(size=0.01)+
  geom_line()+
  #xlab("Number of trajectories added to dataset of size 1,500")+
  xlab("")+
  ylab("")+
  #ylab(expression(paste('Mean  ', f^"CH, *")))+
  #geom_errorbar(aes(ymin=avg+var, ymax=avg-var), width=0.025, alpha=0.5)+
  geom_hline(aes(yintercept=pi_b), linetype=2, color='black')+
  #geom_hline(aes(yintercept=pi_b), linetype=2, color='purple')+
  facet_grid(Estimator+Problem~panaceaAlpha, scales = 'free', labeller = label_parsed, drop=TRUE)+#, labeller = labeller(Estimator = Estimator.labs, panaceaAlpha = panaceaAlpha.labs))+
  #scale_y_continuous(trans = "log")+#breaks=pretty_breaks(4))+#, )+labels=scientific_format(digits=0),
  #scale_y_continuous(trans=weird)+#reverselog_trans(base=10))+
  scale_y_continuous(breaks=pretty_breaks(5))+
  #scale_x_continuous(breaks=seq(0, 100, by=25))+
  #scale_x_continuous(breaks=pretty_breaks(5))+
  #scale_color_discrete(name = 'Estimator')+
  theme(axis.text.x = element_text(angle=45, hjust=1), 
        axis.text=element_text(size=8),
        legend.box.margin = margin(-10,-10,-10,-10), 
        legend.position = "none")+
  scale_color_discrete(name="")+
  ylim(y_min, y_max)

# Generating plots for presentation
# if (WEIGHTING == 'IS') {
#   plot <- plot_grid(p1, p2, align='hv')
#   plot <- plot_grid(plot, legend, rel_widths = c(2.5, .3))
#   pdf(paste0(PATH, WEIGHTING, "grid.pdf"), w=10, h=3)
#   grid.arrange(arrangeGrob(plot))
#   dev.off()
# } else {
#   plot2 <- plot_grid(p1, p2, align='hv')
#   plot2 <- plot_grid(plot2, legend, rel_widths = c(2.5, .3))
#   x.grob <- textGrob("Number of adversarial trajectories added to dataset of size 1,500", gp=gpar(fontsize=14))
#   pdf(paste0(PATH, "grid.pdf"), w=10, h=6)
#   #grid.arrange(arrangeGrob(plot, bottom = x.grob))
#   grid.arrange(arrangeGrob(plot, plot2, bottom = x.grob), ncol=1)
#   dev.off()
# }

# Plots for paper
if (WEIGHTING == 'IS') {
  plot <- plot_grid(p1, p2, align='hv')
  plot <- plot_grid(plot, legend, rel_widths = c(2.5, .3))
  pdf(paste0(PATH, WEIGHTING, "results_final.pdf"), w=10, h=3)
  grid.arrange(arrangeGrob(plot))
  dev.off()
} else {
  plot2 <- plot_grid(p1, p2, align='hv')
  plot2 <- plot_grid(plot2, legend, rel_widths = c(2.5, .3))
  x.grob <- textGrob("Number of adversarial trajectories added to dataset of size 1,500", gp=gpar(fontsize=14))
  pdf(paste0(PATH, "results_final.pdf"), w=10, h=6)
  #grid.arrange(arrangeGrob(plot, bottom = x.grob))
  grid.arrange(arrangeGrob(plot, plot2, bottom = x.grob), ncol=1)
  dev.off()
}

