#!/usr/bin/env ruby
# encoding: utf-8

datasets = ['optdigits', 'mnist', 'fashion_mnist', '20ng', 'svhn']
seeds = [30496, 260819, 240213, 10510, 150364]
d = 50

datasets.each do |dataset|
  seeds.each do |seed|
    puts "python3 ../scripts/run_umap.py ../#{dataset}/#{dataset}_d_#{d}.csv #{seed}"
    puts "python3 ../scripts/run_fitsne.py ../#{dataset}/#{dataset}_d_#{d}.csv #{seed}"
    puts "python3 ../bhtsne.py -i ../#{dataset}/#{dataset}_d_#{d}.csv -o bhtsne_#{dataset}_d_#{d}_s_#{seed}.tsv --randseed #{seed}"
  end
end
