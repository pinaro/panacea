#!/usr/bin/env Rscript
library(dplyr)
library(readr)
library(scales)
library(ggplot2)
library(cowplot)
library(grid)
library(gridExtra)

# Parse command line arguments
args = commandArgs(trailingOnly=TRUE)
J_b_diabetes <- as.double(args[1])
print(J_b_diabetes)
J_e_diabetes <- as.numeric(args[2])
J_b_grid <- as.numeric(args[3])
J_e_grid <- as.numeric(args[4])
Alpha_IS <- c(unlist(strsplit(args[5], ',')))
Alpha_WIS <- c(unlist(strsplit(args[6], ',')))

PATH <- "../results_tmp/"

dat <- read_delim(paste0(PATH, "with_panacea.csv"),
                  delim = '\t',
                  col_types = cols(.default = col_double(), Estimator = col_character(), Problem = col_character()))

dat$pi_b <- 0
dat$pi_e <- 0
dat$pi_b[dat$Problem == 'Diabetes'] <- J_b_diabetes
dat$pi_e[dat$Problem == 'Diabetes'] <- J_e_diabetes
dat$pi_b[dat$Problem == 'Grid-world'] <- J_b_grid
dat$pi_e[dat$Problem == 'Grid-world'] <- J_e_grid

results1 <- dat %>% group_by(k, Estimator, Problem, panaceaAlpha) %>% summarize(avg=mean(Result), var=var(Result), cnt=n(), ci=1.96*sqrt(avg*(1-avg)/n()), pi_e = unique(pi_e), pi_b = unique(pi_b))
# Fix data
results1$Panacea <- 'with\nPanacea'

# Read in data without Panacea
dat <- read_delim(paste0(PATH, "without_panacea.csv"),
                  delim = '\t',
                  col_types = cols(.default = col_double(), Estimator = col_character(), Problem = col_character()))

dat$pi_b <- 0
dat$pi_e <- 0
dat$pi_b[dat$Problem == 'Diabetes'] <- J_b_diabetes
dat$pi_e[dat$Problem == 'Diabetes'] <- J_e_diabetes
dat$pi_b[dat$Problem == 'Grid-world'] <- J_b_grid
dat$pi_e[dat$Problem == 'Grid-world'] <- J_e_grid

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

plot_list = list()
i <- 1
for (WEIGHTING in c('IS', 'WIS')) {
  if (WEIGHTING == 'IS') {
    results_tmp <- results %>% filter(panaceaAlpha %in% as.double(Alpha_IS))
    results_tmp$panaceaAlpha <- factor(results_tmp$panaceaAlpha, levels = Alpha_IS, labels = lapply(Alpha_IS, function(x) paste0(expression(alpha), ': ', x)))
  } else {
    results_tmp <- results %>% filter(panaceaAlpha %in% as.double(Alpha_WIS))
    results_tmp$panaceaAlpha <- factor(results_tmp$panaceaAlpha, levels = Alpha_WIS, labels = lapply(Alpha_WIS, function(x) paste0(expression(alpha), ': ', x)))
  }
  # Combine data frames and plot
  p1 <- ggplot(data=results_tmp %>% filter(Estimator == WEIGHTING))+
    theme_gray(14)+
    aes(x=k, y=avg, group=factor(Panacea), color=factor(Panacea))+
    geom_point(size=0.02)+
    geom_line()+
    xlab("")+
    ylab(expression(paste('Mean  ', L^"CH, *")))+
    geom_hline(aes(yintercept=pi_b), linetype=2, color='black')+
    facet_grid(Estimator+Problem~panaceaAlpha, scales = 'free', labeller = label_parsed)+
    scale_y_continuous(breaks=pretty_breaks(6))+
    scale_x_continuous(breaks=pretty_breaks(5))+
    theme(axis.text.x = element_text(angle=45, hjust=1),
          axis.text=element_text(size=8),
          legend.box.margin = margin(-10,-10,-10,-10),
          legend.position = "right")+
    scale_color_discrete(name="")
  legend <- get_legend(p1 + theme(legend.box.margin = margin(0, 0, 0, 5)))
  p1 <- p1 + theme(legend.position = "none")
  if (WEIGHTING == 'IS') {
    y_min <- -1
    y_max <- 1
  } else {
    y_min <- 0.15
    y_max <-0.9
  }
  p2 <- ggplot(data=results_tmp %>% filter(Estimator == WEIGHTING) %>% filter(Panacea == 'with\nPanacea'))+
    theme_gray(14)+
    aes(x=k, y=avg, group=factor(Panacea), color=factor(Panacea))+
    geom_point(size=0.02)+
    geom_line()+
    xlab("")+
    ylab("")+
    geom_hline(aes(yintercept=pi_b), linetype=2, color='black')+
    facet_grid(Estimator+Problem~panaceaAlpha, scales = 'free', labeller = label_parsed, drop=TRUE)+
    #scale_y_continuous(breaks=pretty_breaks(5))+
    theme(axis.text.x = element_text(angle=45, hjust=1),
          axis.text=element_text(size=8),
          legend.box.margin = margin(-10,-10,-10,-10),
          legend.position = "none")+
    scale_color_discrete(name="")+
    ylim(y_min, y_max)
  
  plot_list[[i]] <- p1
  plot_list[[i+1]] <- p2
  i <- i + 2
}

plot <- plot_grid(plot_list[[1]], plot_list[[2]], align='hv')
plot <- plot_grid(plot, legend, rel_widths = c(2.5, .3))
plot2 <- plot_grid(plot_list[[3]], plot_list[[4]], align='hv')
plot2 <- plot_grid(plot2, legend, rel_widths = c(2.5, .3))
x.grob <- textGrob("Number of adversarial trajectories added to dataset of size 1,500", gp=gpar(fontsize=14))
pdf(paste0(PATH, "results_final.pdf"), w=10, h=6)
grid.arrange(arrangeGrob(plot, plot2, bottom = x.grob), ncol=1)
dev.off()
