# Title     : TODO
# Objective : TODO
# Created by: LH
# Created on: 2022/11/18

library("ggplot2")
library("lattice")
library("ggpubr")
library("svglite")
library(corrplot)

CN_name <- c("ICC")
AD_name <- c("ICC")
iternum1<-vector("double", 100)
iternum2<-vector("double", 100)
for (a in seq_along(iternum1)){
  CN_name[a]<-"ICC"
}
for (b in seq_along(iternum2)){
  AD_name[b]<-"ICC"
}
cfc_ave_normal <- read.table("cfc_ave_normal.txt")
cfc_ave_unnormal <- read.table("cfc_ave_unnormal.txt")
cfc_ave_normal <- as.data.frame(cfc_ave_normal)
cfc_ave_unnormal <- as.data.frame(cfc_ave_unnormal)
cfc_ave_normal <- cor(cfc_ave_normal)
cfc_ave_unnormal <- cor(cfc_ave_unnormal)
test <- diag(10)
cfc_ave_normal <- cfc_ave_normal - test
cfc_ave_unnormal <- cfc_ave_unnormal - test
#col3 <- colorRampPalette(c("red", "white", "blue"))
#corrplot(cfc_ave_normal, method = "shade")# order = "hclust", addrect = 2,, col = col3(20)
#corrplot(cfc_ave_unnormal, method = "shade")# order = "hclust", addrect = 2,, col = col3(20)

#library(rlang)
library(tidyverse)
#require(tidyverse)

# Summarise
my_grouped_summary <-
  function(dataset,
           x,
           y,
           group) {
    quo_x <- sym(x)
    quo_y <- sym(y)
    quo_group <- sym(group)
    #print(dataset)
    summary <- dataset %>% group_by(!!quo_x, !!quo_group) %>%
      summarise(
        sd = sd(!!quo_y, na.rm = T),
        se = sd / sqrt(length(!!quo_y)),
        mean = mean(!!quo_y, na.rm = T),
        n = sum(!is.na(!!quo_y))
      ) %>%
      mutate(
        min = mean - qt(.95, n - 1) * sd/2,
        lower = mean - sd/2,
        upper = mean + sd/2,
        max = mean + qt(.95, n - 1) * sd/2
      )
    return(summary)
  }


# Graph Settings
my_graph_settings <- function(text.angle, text.size) {
  my_graph_settings <-
    theme_bw(base_size = text.size) +
    theme(
      panel.border = element_blank(),
      axis.line = element_line(colour = "black"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.minor.y = element_blank(),
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA),
      legend.key = element_rect(fill = "transparent", colour = NA),
      legend.background = element_rect(fill = "transparent", colour = NA),
      axis.text =  element_text(colour = "black"),
      axis.title = element_text(face = "bold")
    )
  if (text.angle > 0) {
    my_graph_settings <-
      my_graph_settings + theme(axis.text.x = element_text(angle = text.angle, hjust = 1))
  }
  return(my_graph_settings)
}


# Boxplots
my_grouped_boxplot <-
  function(data,
           x,
           y,
           group,
           xlab = x,
           ylab = y,
           width = .5,
           fill = "#2171b5",
           alpha = .7,
           jitter.height = .1,
           points = "dotplot",
           text.angle = 0,
           text.size = 24) {
    quo_x <- sym(x)
    quo_y <- sym(y)
    quo_group <- sym(group)
    summary <- my_grouped_summary(data, x, y, group)
    graph <-
      ggplot(summary, aes_string(x = x,
                                 y = "mean",
                                 #color = group,
                                 fill = group)) +
      geom_boxplot(
        aes_string(
          ymin = "min",
          lower = "lower",
          middle = "mean",
          upper = "upper",
          ymax = "max",
        ),
        #fill = fill,
        width = width,
        position = position_dodge(0.7),
        lwd = 1,
        fatten = 1.5,
        stat = "identity",
        alpha = alpha
      ) +
      ylab(ylab) +
      xlab(xlab) +
      my_graph_settings(text.angle, text.size)
      #ggboxplot(data, x="xlabel", y="icccol", color = "label", palette = "jco", add = "jitter", fill = "label", alpha = 0.8) +stat_compare_means(aes(group=label), label = "p.format")

    if (points == "dotplot") {
      graph <- graph +
        geom_dotplot (
          data = data,# [1:360,]
          aes_string(x = x,
                     y = y,
                     color = group,
                     fill = group
                     ),
          pch = 11,
          color = "black",
          stackdir = "center",
          binaxis = "y",
          position = position_jitterdodge(jitter.width = width / 8, jitter.height = jitter.height/8),#
            #position_dodge(width = width),
          dotsize = .35
        )
    }
      else if (points == "jitter") {
      graph <- graph +
        geom_jitter(
          data = data[1:120,],
          aes_string(x = x,
                     y = y,
                     color = group,
                     fill = group),
          pch = 11,
          color = "black",
          #fill = "gray88",
          alpha = .7,
          position = position_jitterdodge(jitter.width = width / 8,
                                          jitter.height = jitter.height)
        )
    }
    else if (points == "count") {
      graph <- graph +
        geom_count (
          data = data,
          aes_string(x = x,
                     y = y,
                     color = group,
                     fill = group),
          pch = 11,
          #fill = "gray88",
          color = "black",
          position = position_dodge(width = width)
          )
    }
    graph$layers <- rev(graph$layers)

    states_data <- compare_means(icccol~label, data = data, group.by ="xlabel")
    print(states_data)
    print(states_data["p.format"][1,1])
    print(states_data["p.format"][2,1])
    print(states_data["p.format"][3,1])

    graph <- graph + scale_fill_brewer(palette="Set1") + scale_color_brewer(palette="Set1") + ylim(0, 0.5)
    # + theme(legend.position = "none", axis.text.y = element_blank())
    # + coord_fixed(ratio=3)
    #+ geom_signif(data = data,
    #          comparisons=label,
    #          map_signif_level=FALSE,
    #          textsize=5,
    #          test=wilcox.test,
    #          step_increase=0.2,
    #          size = 1)
    #+ stat_compare_means(aes(group=label), label = "p.format")
      #+ scale_fill_manual(values=c("#FF0000", "#00FF00"))

    measurevar <- y
    groupvars  <- c(x, group)
    f <-
      paste(measurevar, paste(groupvars, collapse = " * "), sep = " ~ ")
    lm <- anova(lm(f, data = data))
    filename<-paste("box_plot_ad.svg", sep = "")
    ggsave(filename) # , width = 4.0, height = 3.0
    print(lm)
    print(summary)
    return(graph)
  }


library("R.matlab")
library("ggcor")

#require(ggplot2, quietly = TRUE)

#install.packages("remotes")
#remotes::install_github("houyunhuang/ggcor")
#devtools::install_git("https://gitee.com/houyunhuang/ggcor.git")

#devtools::install_local("C:/Users/LH/Documents/R/win-library/4.0/ggcor_master", force = TRUE, INSTALL_opts="--no-multiarch")

#if(!require(devtools))
#  install.packages("devtools")
#
#if(!require(ggcor))
#  devtools::install_github("houyunhuang/ggcor")
#
#suppressWarnings(suppressMessages(library("ggcor")))

setwd("D:/research/Nonliner Dimensional Reduction/other_work")
path_controlB<-("D:/research/Nonliner Dimensional Reduction/other_work")
controlBname<-file.path(path_controlB, 'adavecfc_order1.mat')# controlB
controlB<-readMat(controlBname)

normfun<-function(data,ymin=0,ymax=1){
  xmax=max(data)
  xmin=min(data)

  y = (ymax-ymin)*(data-xmin)/(xmax-xmin) + ymin

  return(y)

}
#normdata<-normfun(data.frame(controlB[1]))
normdata<-data.frame(controlB[1])
#normdata<-normdata-0.5
#normdata[normdata>1] <- 1
#normdata[normdata<-1] <- -1

graph<-quickcor(normdata, type = "full") + geom_circle2() + remove_axis()
#install.packages("corrplot")
library(corrplot)
g_con<-corrplot(as.matrix(normdata), is.corr = FALSE, method = "circle", tl.pos="n", addgrid.col = "NA")
filename<-paste("adavecfc_order1.png", sep = "")# controlB

svglite(filename, width = 4, height = 4)
g_con<-corrplot(as.matrix(normdata), is.corr = FALSE, method = "circle", tl.pos="n", addgrid.col = "NA")
#dsamp <- diamonds[sample(nrow(diamonds), 1000), ]
#qplot(carat, price, data=dsamp, colour=clarity)
#savePlot(filename = "Rplot",
#         type ="svg",
#         device = dev.cur(),
#         restoreConsole = TRUE)
#library(sjPlot)
#save_plot("your_plot.svg", fig = p, width=10, height=8)
#ggsave(filename, width = 4.0, height = 3.0)


icc_normal <- read.table("mean_normal.txt") # , col.names=CN_name
icc_unnormal <- read.table("mean_unnormal.txt") # , col.names=AD_name
icc_normal_all <- as.data.frame(icc_normal)
icc_unnormal_all <- as.data.frame(icc_unnormal)
icc_normal <- as.data.frame(icc_normal[7:9,])
icc_unnormal <- as.data.frame(icc_unnormal[7:9,])
#icc <- rbind(icc_normal[3,], icc_unnormal[3,], icc_normal[2,], icc_unnormal[2,], icc_normal[1,], icc_unnormal[1,])
#xlabel <- c("first diagonal", "first diagonal", "second diagonal", "second diagonal", "third diagonal", "third diagonal")
xlabel <- c("third diagonal", "second diagonal", "first diagonal")
iternum1<-vector("double", 199)
for (a in seq_along(iternum1)){
  xlabel <- append(xlabel, c("third diagonal", "second diagonal", "first diagonal")) # c(2, 2, 1, 1, 0.5, 0.5)
}
label <- c("CN", "CN", "CN", "AD", "AD", "AD")
iternum2<-vector("double", 99)
for (a in seq_along(iternum2)){
  label <- append(label, c("CN", "CN", "CN", "AD", "AD", "AD"))
}
#library("plyr")
#icc_normal <- cbind(xlabel, icc_normal)
#icc_unnormal <- cbind(xlabel, icc_unnormal)
icc_row <- dplyr::bind_rows(icc_normal, icc_unnormal) # rbind.fill

icccol <- data.frame(v1 = 1:(100 * nrow(icc_row)))
temp1 <- icc_row[,1]
for (i in 1:(ncol(icc_row)/100)) {
  for (j in 2:100) {
    temp1 <- append(temp1, abs(c(icc_row[,j])))
  }
}

icccol <- temp1

icc <- cbind(icccol, label)
icc <- cbind(icc, xlabel)
icc <- data.frame(icc)
#icc %>% select_if(~!any(is.na(.)))


icc %>% mutate(label = factor(label, labels = c("CN", "AD")), xlabel = factor(xlabel), icccol = as.numeric(icccol)) %>% my_grouped_boxplot(x = "xlabel", y = "icccol", group = "label", xlab = "Different diagonal", ylab = "ICC") + labs(fill = "Disease")

#ToothGrowth %>% mutate(supp = factor(supp, labels = c("Orange Juice", "Ascorbic Acid")), dose = factor(dose)) %>% my_grouped_boxplot(x = "dose", y = "len", group = "supp", xlab = "Vitamin C Dose (mg/day)", ylab = "Tooth Length") + labs(fill = "Delivery Method")
summary(icc)
#summary(ToothGrowth)

## bk in task
#icc_0bk <- read.table("icc_0bk.txt") # , col.names=CN_name
#icc_2bk <- read.table("icc_2bk.txt") # , col.names=AD_name
#icc_0bk_all <- as.data.frame(icc_0bk)
#icc_2bk_all <- as.data.frame(icc_2bk)
#icc_0bk <- as.data.frame(icc_0bk[7:9,])
#icc_2bk <- as.data.frame(icc_2bk[7:9,])
#xlabel <- c("third diagonal", "second diagonal", "first diagonal")
#iternum1<-vector("double", 199)
#for (a in seq_along(iternum1)){
#  xlabel <- append(xlabel, c("third diagonal", "second diagonal", "first diagonal")) # c(2, 2, 1, 1, 0.5, 0.5)
#}
#label <- c("0bk", "0bk", "0bk", "2bk", "2bk", "2bk")
#iternum2<-vector("double", 99)
#for (a in seq_along(iternum2)){
#  label <- append(label, c("0bk", "0bk", "0bk", "2bk", "2bk", "2bk"))
#}
#icc_row <- dplyr::bind_rows(icc_0bk, icc_2bk) # rbind.fill
#
#icccol <- data.frame(v1 = 1:(100 * nrow(icc_row)))
#temp1 <- icc_row[,1]
#for (i in 1:(ncol(icc_row)/100)) {
#  for (j in 2:100) {
#    temp1 <- append(temp1, abs(c(icc_row[,j])))
#  }
#}
#
#icccol <- temp1
#
#icc <- cbind(icccol, label)
#icc <- cbind(icc, xlabel)
#icc <- data.frame(icc)
#
#icc %>% mutate(label = factor(label, labels = c("0bk", "2bk")), xlabel = factor(xlabel), icccol = as.numeric(icccol)) %>% my_grouped_boxplot(x = "xlabel", y = "icccol", group = "label", xlab = "Different diagonal", ylab = "ICC") + labs(fill = "Disease")


## sub in task
#icc_place <- read.table("icc_place.txt")
#icc_body <- read.table("icc_body.txt")
#icc_face <- read.table("icc_face.txt")
#icc_tools <- read.table("icc_tools.txt")
#icc_place_all <- as.data.frame(icc_place)
#icc_body_all <- as.data.frame(icc_body)
#icc_face_all <- as.data.frame(icc_face)
#icc_tools_all <- as.data.frame(icc_tools)
#icc_place <- as.data.frame(icc_place[7:9,])
#icc_body <- as.data.frame(icc_body[7:9,])
#icc_face <- as.data.frame(icc_face[7:9,])
#icc_tools <- as.data.frame(icc_tools[7:9,])
#xlabel <- c("third diagonal", "second diagonal", "first diagonal")
#iternum1<-vector("double", 399)
#for (a in seq_along(iternum1)){
#  xlabel <- append(xlabel, c("third diagonal", "second diagonal", "first diagonal")) # c(2, 2, 1, 1, 0.5, 0.5)
#}
#label <- c("place", "place", "place", "body", "body", "body", "face", "face", "face", "tools", "tools", "tools")
#iternum2<-vector("double", 99)
#for (a in seq_along(iternum2)){
#  label <- append(label, c("place", "place", "place", "body", "body", "body", "face", "face", "face", "tools", "tools", "tools"))
#}
#icc_row <- dplyr::bind_rows(icc_place, icc_body, icc_face, icc_tools) # rbind.fill
#
#icccol <- data.frame(v1 = 1:(100 * nrow(icc_row)))
#temp1 <- icc_row[,1]
#for (i in 1:(ncol(icc_row)/100)) {
#  for (j in 2:100) {
#    temp1 <- append(temp1, abs(c(icc_row[,j])))
#  }
#}
#
#icccol <- temp1
#
#icc <- cbind(icccol, label)
#icc <- cbind(icc, xlabel)
#icc <- data.frame(icc)
#
#icc %>% mutate(label = factor(label, labels = c("place", "body", "face", "tools")), xlabel = factor(xlabel), icccol = as.numeric(icccol)) %>% my_grouped_boxplot(x = "xlabel", y = "icccol", group = "label", xlab = "Different diagonal", ylab = "ICC") + labs(fill = "Disease")

