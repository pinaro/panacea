library(dplyr)
library(readr)
library(scales)
library(ggplot2)

path <- "/Users/Pinar/Desktop/"

dat <- read_delim(paste0(path,"diabetes_results.csv"),
                  delim = '\t', 
                  col_types = cols(.default = col_double(), Estimator = col_character()))
dat$PassOrFail <- dat$Result > 0.21769472415028138

results <- dat %>% group_by(k, Estimator) %>% summarize(avg=mean(Result), var=var(Result), cnt=n(), ci=1.96*sqrt(avg*(1-avg)/n()))
#results <- results %>% filter(Estimator == "CH, IS" | Estimator == "AM, IS")
 
results <- dat %>% group_by(k, Estimator) %>% summarize(avg=mean(PassOrFail==TRUE),var=var(Result), cnt=n(), ci=1.96*sqrt(avg*(1-avg)/n()))
results <- results %>% filter(Estimator == "AM, WIS")

pdf(paste0(path,"diabetes.pdf"),w=5,h=4)
ggplot(data=results)+ #
  theme_gray(14)+
  #theme(text = element_text(size=12), axis.text.x = element_text(angle=45, hjust=1))+
  aes(x=k, y=avg, group=Estimator, color=Estimator)+
  geom_point(size=0.2)+
  #geom_jitter(size=0.5)+
  geom_line()+
  xlab("Number of adversarial trajectories \n added to dataset of size 1,500")+
  ylab("Probability of undesirable policy\n passing safety test")+
  geom_errorbar(aes(ymin=avg+ci, ymax=avg-ci), width=0.025, alpha=0.5)+
  #geom_hline(aes(yintercept=0.21756437864732214), linetype=2, color='black')+
  #facet_wrap(N~., scales = 'free', labeller = label_both)+
  #scale_y_continuous(breaks=pretty_breaks(8))+
  scale_x_continuous(breaks=seq(0, 100, by=10))+
  #scale_x_continuous(breaks=pretty_breaks(5))+
  #scale_color_discrete(name = 'Estimator')+
  theme(legend.position="top",
        legend.box.margin = margin(-10,-10,-10,-10),
        legend.title = element_blank())
dev.off()
