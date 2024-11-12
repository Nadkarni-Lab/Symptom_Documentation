library(readxl)
library(ggplot2)

symp_data <- read_excel("./data/symptom_data.xls")
data <- symp_data[, -c(2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,45)]
dat <- data.matrix(data)
data[dat == 101] <- 61
data[dat == 100] <- 58
data[dat == 99] <- 29
data[dat == 98] <- 13


num_symp <- ncol(data)


perc_tot <- c()
for (j in 2:num_symp) {
  count <- 0
  for(i in 1:97) {
    x <- subset(dat, data['studyid'] == i)
    y <- x[,j]
    sum_ <- sum(y)
    if(sum_ > 0) {
      count = count + 1
    }
  }
  perc <- (count / 97)*100
  perc_tot <- c(perc_tot, perc)
}

print(perc_tot)

b <- c('Constipation','Nausea','Vomiting','Diarrhea','Appetite','Cramp','Edema',
       'SOB','Dizzy','RLS','Fatigue','Cough','Dry mouth','Bone pain','Chest pain',
       'Headache','Muscle soreness','Concentration','Dry skin','Itching','IDH')
df <- data.frame(value = perc_tot,Symptom = b)
p<-ggplot(data=df, aes(x= reorder(Symptom,-value),y = value)) + scale_y_continuous(limits = c(0,63), expand = c(0, 0), breaks = seq(0,60, by=10)) + theme_bw() + xlab("Symptom") +
  geom_bar(stat="identity", fill = 'gray30', width = 0.6, colour = 'black', size = 0.4) + ylab("Percent of Patients Reporting") + ggtitle("Proportion of Patients Reporting Symptoms During Study Period") +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5),axis.text.x = element_text(size = 12,angle = 90, vjust = 0.5, hjust=1, face = "bold"), panel.border = element_blank(),panel.grid.major.x = element_blank(),panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 12),
                     panel.grid.minor.x = element_blank(), panel.grid.minor.y = element_blank(), axis.line = element_line(colour = "gray30"),  axis.title.x = element_text(size = 18), axis.title.y = element_text(size = 18))
p
