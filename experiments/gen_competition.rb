#!/usr/bin/env ruby
# encoding: utf-8

# datasets = ['dblp', 'cora', 'citeseer', 'emailEuCore', 'pubmed', 'optdigits']
datasets = ["fashion_mnist", "mnist", "20ng", "optdigits", "svhn"]
seeds = [30496, 260819, 240213, 10510, 150364]
d = 50

datasets.each do |dataset|
  seeds.each do |seed|
    puts "python3 ../scripts/run_umap.py ../data/#{dataset}/#{dataset}_d_#{d}.csv #{seed}"
    puts "python3 ../scripts/run_fitsne.py ../data/#{dataset}/#{dataset}_d_#{d}.csv #{seed}"
    puts "python3 ../bhtsne/bhtsne.py -i ../data/#{dataset}/#{dataset}_d_#{d}.csv -o bhtsne_#{dataset}_d_#{d}_s_#{seed}.tsv --randseed #{seed} --no_pca"
    puts "../ktsne ../data/#{dataset}/#{dataset}_d_#{d}.csv -p 50 -n 200 -i 1000 -t 50 -s #{seed}"
  end
end
