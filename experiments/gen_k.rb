#!/usr/bin/env ruby
# encoding: utf-8

datasets = ['optdigits', 'mnist']
seeds = [30496, 260819, 240213, 10510, 150364]
ks = [5, 10, 30, 50, 100]

datasets.each do |dataset|
  seeds.each do |seed|
    ks.each do |k|
      puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv -k #{k} -K #{k} -s #{seed}"
    end

    puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv -k 5 -K 50 -s #{seed}"
  end
end
