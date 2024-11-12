install.packages("RColorBrewer")
library(ggplot2)
# library(tidyverse)C

data <- read.csv("./data/by_person.csv")
data_race <- read.csv("./data/race.csv")
repeat_filt <- is.na(data_race$redcap_repeat_instance)
study_ids <- data_race$studyid[repeat_filt]
race <- data_race$race[repeat_filt]
race_other <- data_race$otherrace[repeat_filt]
hispanic <- data_race$hispanic[repeat_filt]

pat_data <- subset(data, data$redcap_repeat_instrument == "dsi")
patient <- pat_data[, -c(3:14,16:24,26:36,38:40,42,44,46:103)]

nurse_data <- subset(data, data$redcap_repeat_instrument == "nurse")
nurse <- nurse_data[, -c(3:75,77:80,82:86,88,92:103)]

doc_data <- subset(data, data$redcap_repeat_instrument == "")
doc <- doc_data[, -c(3:52,54:57,59:63,65,69:103)]

perc_tot_n <- c()
race0_perc_tot_n <- c()
race1_perc_tot_n <- c()
race2_perc_tot_n <- c()
race6_perc_tot_n <- c()
for (i in 3:7) {
  count_n <- 0
  race0_count_n <- 0
  race1_count_n <- 0
  race2_count_n <- 0
  race6_count_n <- 0
  for (j in 1:97) {
    x_n <- subset(nurse, nurse['studyid'] == j)
    y_n <- x_n[,i]
    sum_n <- sum(y_n)
    if (sum_n > 0) {
      count_n = count_n + 1
      if (race[j] == 0) {
        race0_count_n = race0_count_n + 1
      } else if (race[j] == 1) {
        race1_count_n = race1_count_n + 1
      } else if (race[j] == 2) {
        race2_count_n = race2_count_n + 1
      } else if (race[j] == 6) {
        race6_count_n = race6_count_n + 1
      }
    }
  }
  perc_n <- (count_n / 97)*100
  perc_tot_n <- c(perc_tot_n, perc_n)
  race0_perc_tot_n <- c(race0_perc_tot_n, (100*race0_count_n / sum(race[1:97]==0)))
  race1_perc_tot_n <- c(race1_perc_tot_n, (100*race1_count_n / sum(race[1:97]==1)))
  race2_perc_tot_n <- c(race2_perc_tot_n, (100*race2_count_n / sum(race[1:97]==2)))
  race6_perc_tot_n <- c(race6_perc_tot_n, (100*race6_count_n / sum(race[1:97]==6)))
}

perc_tot_p <- c()
race0_perc_tot_p <- c()
race1_perc_tot_p <- c()
race2_perc_tot_p <- c()
race6_perc_tot_p <- c()
for (i in 3:7) {
  count_p <- 0
  race0_count_p <- 0
  race1_count_p <- 0
  race2_count_p <- 0
  race6_count_p <- 0
  for (j in 1:97) {
    x_p <- subset(patient, patient['studyid'] == j)
    y_p <- x_p[,i]
    sum_p <- sum(y_p)
    if (sum_p > 0) {
      count_p = count_p + 1
      if (race[j] == 0) {
        race0_count_p = race0_count_p + 1
      } else if (race[j] == 1) {
        race1_count_p = race1_count_p + 1
      } else if (race[j] == 2) {
        race2_count_p = race2_count_p + 1
      } else if (race[j] == 6) {
        race6_count_p = race6_count_p + 1
      }
    } 
  }
  perc_p <- (count_p / 97)*100
  perc_tot_p <- c(perc_tot_p, perc_p)
  race0_perc_tot_p <- c(race0_perc_tot_p, (100*race0_count_p / sum(race[1:97]==0)))
  race1_perc_tot_p <- c(race1_perc_tot_p, (100*race1_count_p / sum(race[1:97]==1)))
  race2_perc_tot_p <- c(race2_perc_tot_p, (100*race2_count_p / sum(race[1:97]==2)))
  race6_perc_tot_p <- c(race6_perc_tot_p, (100*race6_count_p / sum(race[1:97]==6)))
}

perc_tot_d <- c()
race0_perc_tot_d <- c()
race1_perc_tot_d <- c()
race2_perc_tot_d <- c()
race6_perc_tot_d <- c()
for (i in 3:7) {
  count_d <- 0
  race0_count_d <- 0
  race1_count_d <- 0
  race2_count_d <- 0
  race6_count_d <- 0
  for (j in 1:97) {
    x_d <- subset(doc, doc['studyid'] == j)
    y_d <- x_d[,i]
    sum_d <- sum(y_d)
    if (sum_d > 0) {
      count_d = count_d + 1
      if (race[j] == 0) {
        race0_count_d = race0_count_d + 1
      } else if (race[j] == 1) {
        race1_count_d = race1_count_d + 1
      } else if (race[j] == 2) {
        race2_count_d = race2_count_d + 1
      } else if (race[j] == 6) {
        race6_count_d = race6_count_d + 1
      }
    }
  }
  perc_d <- (count_d / 97)*100
  perc_tot_d <- c(perc_tot_d, perc_d)
  race0_perc_tot_d <- c(race0_perc_tot_d, (100*race0_count_d / sum(race[1:97]==0)))
  race1_perc_tot_d <- c(race1_perc_tot_d, (100*race1_count_d / sum(race[1:97]==1)))
  race2_perc_tot_d <- c(race2_perc_tot_d, (100*race2_count_d / sum(race[1:97]==2)))
  race6_perc_tot_d <- c(race6_perc_tot_d, (100*race6_count_d / sum(race[1:97]==6)))
}


b <- c('Cramp', 'Fatigue', 'Muscle \nsoreness', 'Dry skin', 'Itching') # , 'IDH'
df <- data.frame(value = perc_tot_n, race0 = race0_perc_tot_n, 
                 race1 = race1_perc_tot_n, race2 = race2_perc_tot_n, 
                 race6 = race6_perc_tot_n, Symptom = b)
df$Method <- "Nurse Survey"

# b <- c('Cramp', 'Fatigue', 'Muscle \nsoreness', 'Dry skin', 'Itching', 'IDH')
df1 <- data.frame(value = perc_tot_p, race0 = race0_perc_tot_p, 
                  race1 = race1_perc_tot_p, race2 = race2_perc_tot_p, 
                  race6 = race6_perc_tot_p, Symptom = b)
df1$Method <- "Patient Survey"

# b <- c('Cramp', 'Fatigue', 'Muscle \nsoreness', 'Dry skin', 'Itching', 'IDH')
df2 <- data.frame(value = perc_tot_d, race0 = race0_perc_tot_d, 
                  race1 = race1_perc_tot_d, race2 = race2_perc_tot_d, 
                  race6 = race6_perc_tot_d, Symptom = b)
df2$Method <- "Physician Survey"

# b <- c('Fatigue','Cramp','Dry skin','Muscle \nsoreness','Itching','IDH')
# nlp_perc <- c((1/97)*100,(14/97)*100,0,(5/97)*100,(15/97)*100,(8/97)*100)
# df3 <- data.frame(value = nlp_perc, Symptom = b)
# df3$Method <- "NLP of EHR"

total <- rbind(df1,df2)
total$Method <- factor(total$Method, levels = c("Patient Survey", "Physician Survey"))

level_order <- c('Fatigue', 'Cramp', 'Dry skin', 'Muscle \nsoreness', 'Itching') # 'IDH'
p <- ggplot(data=total, aes(x= factor(Symptom,level = level_order),y = value, fill=Method)) +
  geom_bar(stat="identity", width = 0.6,position=position_dodge(width=0.6),colour = 'black', size = 0.4)+ scale_y_continuous(limits = c(0,66), expand = c(0, 0), breaks = seq(0,65, by=10)) + theme_bw() +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5), axis.text.x = element_text(size = 18, face = "bold"),panel.border = element_blank(),panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),axis.text.y = element_text(size = 16),
        panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(),axis.line = element_line(colour = "gray30"),  axis.title.x = element_text(size = 20), axis.title.y = element_text(size = 18),legend.position = c(0.9, 0.8),legend.title=element_text(size=18), 
        legend.text=element_text(size=16)) + xlab('Symptom') +
  ylab("Percent of Patients Reported with Symptom") + ggtitle("Symptom Prevalence Reported by Patients, Provider Surveys, and EHR") + scale_fill_manual(values=c('midnightblue','grey','lightpink3','wheat2')) 
p

png(file="./figures/symptoms_byrace_black.png",
    width=800, height=600)
p0 <- ggplot(data=total, aes(x= factor(Symptom,level = level_order), y = race0, fill=Method)) +
  geom_bar(stat="identity", width = 0.6,position=position_dodge(width=0.6),colour = 'black', size = 0.4)+ scale_y_continuous(limits = c(0,66), expand = c(0, 0), breaks = seq(0,65, by=10)) + theme_bw() +
  theme(plot.title = element_text(size = 22, face = "bold", hjust = 0.5), 
        axis.text.x = element_text(size = 24, face = "bold"),panel.border = element_blank(),panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = 24),
        panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(),axis.line = element_line(colour = "gray30"),  
        axis.title.x = element_text(size = 24), 
        axis.title.y = element_text(size = 24),legend.position = c(0.8, 0.85),
        legend.title=element_text(size=22), 
        legend.text=element_text(size=20)) + xlab('Symptom') +
  ylab("Percent of Patients Reported with Symptom") + ggtitle("Symptom Prevalence Reported by \nPatients and Provider Surveys (Black)") + 
  scale_fill_manual(values=c('midnightblue','grey','lightpink3','wheat2')) + 
  theme(plot.title = element_text(size=28))
p0
dev.off()

png(file="./figures/symptoms_byrace_white.png",
    width=800, height=600)
p1 <- ggplot(data=total, aes(x= factor(Symptom,level = level_order), y = race1, fill=Method)) +
  geom_bar(stat="identity", width = 0.6,position=position_dodge(width=0.6),colour = 'black', size = 0.4)+ scale_y_continuous(limits = c(0,66), expand = c(0, 0), breaks = seq(0,65, by=10)) + theme_bw() +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5), 
        axis.text.x = element_text(size = 24, face = "bold"),panel.border = element_blank(),panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = 24), panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(),axis.line = element_line(colour = "gray30"),  
        axis.title.x = element_text(size = 24), 
        axis.title.y = element_text(size = 24),legend.position = c(0.8, 0.85),
        legend.title=element_text(size=22), 
        legend.text=element_text(size=20)) + xlab('Symptom') +
  ylab("Percent of Patients Reported with Symptom") + ggtitle("Symptom Prevalence Reported by \nPatients and Provider Surveys (White)") + scale_fill_manual(values=c('midnightblue','grey','lightpink3','wheat2')) + 
  theme(plot.title = element_text(size=28))
p1
dev.off()

p2 <- ggplot(data=total, aes(x= factor(Symptom,level = level_order), y = race2, fill=Method)) +
  geom_bar(stat="identity", width = 0.6,position=position_dodge(width=0.6),colour = 'black', size = 0.4)+ scale_y_continuous(limits = c(0,66), expand = c(0, 0), breaks = seq(0,65, by=10)) + theme_bw() +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5), axis.text.x = element_text(size = 18, face = "bold"),panel.border = element_blank(),panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),axis.text.y = element_text(size = 16),
        panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(),axis.line = element_line(colour = "gray30"),  axis.title.x = element_text(size = 20), axis.title.y = element_text(size = 18),legend.position = c(0.9, 0.8),legend.title=element_text(size=18), 
        legend.text=element_text(size=16)) + xlab('Symptom') +
  ylab("Percent of Patients Reported with Symptom") + ggtitle("Symptom Prevalence Reported by \nPatients and Provider Surveys (Asian)") + scale_fill_manual(values=c('midnightblue','grey','lightpink3','wheat2')) 
p2

png(file="./figures/symptoms_byrace_hispanic.png",
    width=800, height=600)
p6 <- ggplot(data=total, aes(x= factor(Symptom,level = level_order), y = race6, fill=Method)) +
  geom_bar(stat="identity", width = 0.6,position=position_dodge(width=0.6),colour = 'black', size = 0.4)+ scale_y_continuous(limits = c(0,66), expand = c(0, 0), breaks = seq(0,65, by=10)) + theme_bw() +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5), 
        axis.text.x = element_text(size = 24, face = "bold"),panel.border = element_blank(),panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size = 24), panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(),axis.line = element_line(colour = "gray30"),  
        axis.title.x = element_text(size = 24), 
        axis.title.y = element_text(size = 24),legend.position = c(0.8, 0.85),
        legend.title=element_text(size=22), 
        legend.text=element_text(size=20)) + xlab('Symptom') +
  ylab("Percent of Patients Reported with Symptom") + ggtitle("Symptom Prevalence Reported by \nPatients and Provider Surveys (Hispanic)") + scale_fill_manual(values=c('midnightblue','grey','lightpink3','wheat2')) + 
  theme(plot.title = element_text(size=28))
p6
dev.off()
