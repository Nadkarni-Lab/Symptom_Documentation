install.packages("RColorBrewer")
library(ggplot2)

data <- read.csv("./data/by_person.csv")

pat_data <- subset(data, data$redcap_repeat_instrument == "dsi")
patient <- pat_data[, -c(3:14,16:24,26:36,38:40,42,44,46:103)]

nurse_data <- subset(data, data$redcap_repeat_instrument == "nurse")
nurse <- nurse_data[, -c(3:75,77:80,82:86,88,92:103)]

doc_data <- subset(data, data$redcap_repeat_instrument == "")
doc <- doc_data[, -c(3:52,54:57,59:63,65,69:103)]

perc_tot_n <- c()
for (i in 3:8) {
  count_n <- 0
  for (j in 1:97) {
    x_n <- subset(nurse, nurse['studyid'] == j)
    y_n <- x_n[,i]
    sum_n <- sum(y_n)
    if (sum_n > 0) {
      count_n = count_n + 1
    }
  }
  perc_n <- (count_n / 97)*100
  perc_tot_n <- c(perc_tot_n, perc_n)
}

perc_tot_p <- c()
for (i in 3:8) {
  count_p <- 0
  for (j in 1:97) {
    x_p <- subset(patient, patient['studyid'] == j)
    y_p <- x_p[,i]
    sum_p <- sum(y_p)
    if (sum_p > 0) {
      count_p = count_p + 1
    }
  }
  perc_p <- (count_p / 97)*100
  perc_tot_p <- c(perc_tot_p, perc_p)
}

perc_tot_d <- c()
for (i in 3:8) {
  count_d <- 0
  for (j in 1:97) {
    x_d <- subset(doc, doc['studyid'] == j)
    y_d <- x_d[,i]
    sum_d <- sum(y_d)
    if (sum_d > 0) {
      count_d = count_d + 1
    }
  }
  perc_d <- (count_d / 97)*100
  perc_tot_d <- c(perc_tot_d, perc_d)
}


b <- c('Cramp', 'Fatigue', 'Muscle \nsoreness', 'Dry skin', 'Itching', 'IDH')
df <- data.frame(value = perc_tot_n,Symptom = b)
df$Method <- "Nurse Survey"

b <- c('Cramp', 'Fatigue', 'Muscle \nsoreness', 'Dry skin', 'Itching', 'IDH')
df1 <- data.frame(value = perc_tot_p,Symptom = b)
df1$Method <- "Patient Survey"

b <- c('Cramp', 'Fatigue', 'Muscle \nsoreness', 'Dry skin', 'Itching', 'IDH')
df2 <- data.frame(value = perc_tot_d,Symptom = b)
df2$Method <- "Physician Survey"

b <- c('Fatigue','Cramp','Dry skin','Muscle \nsoreness','Itching','IDH')
nlp_perc <- c((1/97)*100,(14/97)*100,0,(5/97)*100,(15/97)*100,(8/97)*100)
df3 <- data.frame(value = nlp_perc, Symptom = b)
df3$Method <- "NLP of EHR"

total <- rbind(df,df1,df2,df3)
total$Method <- factor(total$Method, levels = c("Patient Survey", "Nurse Survey", "Physician Survey", "NLP of EHR"))

level_order <- c('Fatigue', 'Cramp', 'Dry skin', 'Muscle \nsoreness', 'Itching', 'IDH')
p <- ggplot(data=total, aes(x= factor(Symptom,level = level_order),y = value, fill=Method)) +
  geom_bar(stat="identity", width = 0.6,position=position_dodge(width=0.6),colour = 'black', size = 0.4)+ scale_y_continuous(limits = c(0,66), expand = c(0, 0), breaks = seq(0,65, by=10)) + theme_bw() +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5), axis.text.x = element_text(size = 18, face = "bold"),panel.border = element_blank(),panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),axis.text.y = element_text(size = 16),
        panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(),axis.line = element_line(colour = "gray30"),  axis.title.x = element_text(size = 20), axis.title.y = element_text(size = 18),legend.position = c(0.9, 0.8),legend.title=element_text(size=18), 
        legend.text=element_text(size=16)) + xlab('Symptom') +
  ylab("Percent of Patients Reported with Symptom") + ggtitle("Symptom Prevalence Reported by Patients, Provider Surveys, and EHR") + scale_fill_manual(values=c('midnightblue','grey','lightpink3','wheat2')) 
p
