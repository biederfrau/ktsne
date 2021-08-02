#!/usr/bin/env ruby
# encoding: utf-8

datasets = ['optdigits', 'mnist']
seeds = [30496, 260819, 240213, 10510, 150364]
inits = ['pca', 'random']

datasets.each do |dataset|
  seeds.each do |seed|
    puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv --pca-init -k 50 -K 50 -s #{seed}"
    puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv --random-init -k 50 -K 50 -s #{seed}"
    puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv --pca-init -k 30 -K 30 -s #{seed}"
    puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv --random-init -k 30 -K 30 -s #{seed}"
  end
end
