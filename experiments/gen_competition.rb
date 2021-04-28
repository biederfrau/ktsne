#!/usr/bin/env ruby
# encoding: utf-8

datasets = [
  # 'cifar',
  # 'coil-20',
  # 'dblp',
  # 'cora',
  # 'citeseer',
  # 'emailEuCore',
  # 'pubmed',
  # 'optdigits',
  # 'fashion_mnist',
  # 'mnist',
  'svhn',
  # 'timit',
  # 'com-amazon',
  # 'com-dblp',
  # 'hollywood',
  # 'wiki-topcats',
  # 'youtube'
]

datasets = [
  "random_n_100_64.csv",
  "random_n_500_64.csv",
  "random_n_1000_64.csv",
  "random_n_5000_64.csv",
  "random_n_10000_64.csv",
  "random_n_50000_64.csv",
  "random_n_100000_64.csv",
  "random_n_500000_64.csv",
  "random_n_1000000_64.csv",
  #"random_n_5000000_64.csv"
]

seeds = [30496, 260819, 240213, 10510, 150364]
d = 50

datasets.each do |dataset|
  seeds.each do |seed|
    # puts "python3 ../scripts/run_umap.py ../data/#{dataset}/#{dataset}_d_#{d}.csv #{seed}"
    # puts "python3 ../scripts/run_fitsne.py ../data/#{dataset}/#{dataset}_d_#{d}.csv #{seed}"
    # puts "python3 ../bhtsne/bhtsne.py -i ../data/#{dataset}/#{dataset}_d_#{d}.csv -o bhtsne_#{dataset}_d_#{d}_s_#{seed}.tsv --randseed #{seed} --no_pca -p 30"
    # puts "../ktsne ../data/#{dataset}/#{dataset}_d_#{d}.csv -s #{seed}"

    dataset = File.basename dataset, File.extname(dataset)
    puts "python3 ../scripts/run_umap.py ../data/random/#{dataset}.csv #{seed}"
    puts "python3 ../scripts/run_fitsne.py ../data/random/#{dataset}.csv #{seed}"
    # puts "python3 ../bhtsne/bhtsne.py -i ../data/random/#{dataset}.csv -o bhtsne_#{dataset}_s_#{seed}.tsv --randseed #{seed} --no_pca -p 30"
    # puts "../ktsne ../data/random/#{dataset}.csv -s #{seed}"
  end
end
