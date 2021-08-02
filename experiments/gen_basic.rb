#!/usr/bin/env ruby
# encoding: utf-8

datasets = ['optdigits', 'mnist']
etas     = [0.1, 10, 20, 50, 100, 200, 400]
perps    = [3, 5, 15, 30, 50]
seeds = [30496, 260819, 240213, 10510, 150364]

datasets.each do |dataset|
  seeds.each do |seed|
    etas.each do |eta|
      perps.each do |perp|
        puts "../ktsne -n #{eta} -p #{perp} -k 30 -K 30 ../data/#{dataset}/#{dataset}_d_50.csv -s #{seed}"
      end
    end
  end
end
