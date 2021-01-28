#!/usr/bin/env ruby
# encoding: utf-8

datasets = ['optdigits', 'mnist_d_64']
etas     = [0.1, 5, 10, 20, 50, 100, 200]
perps    = [3, 5, 15, 30, 50]

eta  = 20
perp = 5

datasets.each do |dataset|
  dataset_plain = dataset.split('_').first
  etas.each do |eta|
    puts "../ktsne -n #{eta} -p #{perp} -i 1000 -o ../data/#{dataset_plain}/#{dataset}.csv ../data/#{dataset_plain}/#{dataset_plain}_labels.txt > #{dataset}_obj_eta_#{eta}_p_#{perp}.csv"
    puts "mv #{dataset}_embedding.csv ../emb/#{dataset}_embedding_eta_#{eta}_p_#{perp}.csv"
  end

  perps.each do |perp|
    puts "../ktsne -n #{perp} -p #{perp} -i 1000 -o ../data/#{dataset_plain}/#{dataset}.csv ../data/#{dataset_plain}/#{dataset.split('_')[0]}_labels.txt > #{dataset}_obj_eta_#{eta}_p_#{perp}.csv"
    puts "mv #{dataset}_embedding.csv ../emb/#{dataset}_embedding_eta_#{eta}_p_#{perp}.csv"
  end
end
