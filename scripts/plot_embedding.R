#!/usr/bin/env Rscript

library(ggplot2)
library(ggthemes)
library(scales)
library(tools)

theme_set(theme_solid())

argv <- commandArgs(trailingOnly=T)

f <- argv[1]
s <- strsplit(basename(tools::file_path_sans_ext(f)), '_')[[1]]

algo <- s[1]
dataset <- s[2]
sz <- 0.8

# ugh the regex i used in python for this does not work the same
# in R for whatever reason
if(dataset == "fashion") {
    dataset <- paste(s[2], '_', s[3], sep='')
}

folder <- dataset

if(dataset == 'syn2') {
    dataset <- paste(s[2], '_', s[3], '_', s[4], sep='')
    folder <- 'syn2'
}

df <- read.csv(f, header=!grepl("bhtsne", f))
colnames(df) <- c("x", "y")

labels.path <- file.path(getwd(), '..', 'data', folder, paste(dataset, "_labels.txt", sep=""))
print(labels.path)
if(file.exists(labels.path)) {
    print('has labels')
    labels <- read.csv(labels.path, header=F)

    df$label <- as.factor(labels$V1)
    pal <- tableau_color_pal('Superfishel Stone')(nlevels(df$label))
    pal <- rep(na.omit(pal), nlevels(df$label))

    ggplot(df, aes(x=x, y=y, color=label)) + geom_point(show.legend=F, size=sz) + scale_color_manual(values=pal)
} else {
    print('has no labels')
    ggplot(df, aes(x=x, y=y)) + geom_point(show.legend=F, size=sz, color="darkgray")
}

outname <- paste(tools::file_path_sans_ext(f), ".png", sep="")
ggsave(outname, device="png", width=30, height=30, units="cm", dpi="print")

print(paste("output in", outname))
