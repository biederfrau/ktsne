#!/usr/bin/env ruby
# encoding: utf-8

datasets = ['optdigits', 'mnist']
exagg = [1, 1.5, 4, 8, 12]
seeds = [30496, 260819, 240213, 10510, 150364]

datasets.each do |dataset|
  exagg.each do |x|
    exagg.each do |xx|
      seeds.each do |seed|
        puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv -x #{x} -X #{xx} -s #{seed}"
      end
    end
  end
end
