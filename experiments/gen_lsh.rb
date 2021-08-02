#!/usr/bin/env ruby
# encoding: utf-8

datasets = ['optdigits', 'mnist']
ls = [10, 20, 50, 100]
seeds = [30496, 260819, 240213, 10510, 150364]

datasets.each do |dataset|
  ls.each do |l|
    seeds.each do |seed|
      puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv -k 30 -K 30 --num-hash-tables #{l} --use-cross-polytope -s #{seed}"
      puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv -k 30 -K 30 --num-hash-tables #{l} --use-hyperplane -s #{seed}"
    end
  end
end
