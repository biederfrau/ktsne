#!/usr/bin/env Rscript

library(ggplot2)
library(ggthemes)
library(tools)

theme_set(theme_light())

f <- commandArgs(trailingOnly=T)[1]
m <- regmatches(f, regexec(".*/(.*)_embedding_it_(\\d+).csv", f))[[1]]

if(is.na(m[2])) {
    m <- regmatches(f, regexec("(.*)_embedding.*.csv", f))[[1]]
}

df <- read.csv(f)

df$label <- as.factor(df$label)
ggplot(df, aes(x=x, y=y)) + geom_point(aes(color=label)) + ggtitle(paste(m[2], ", it = ", m[3], sep=""))

ggsave(paste(tools::file_path_sans_ext(f), ".jpg", sep=""), device="jpg", width=10, height=7.5, units="cm", dpi="print")
