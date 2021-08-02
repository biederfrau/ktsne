#!/usr/bin/env ruby
# encoding: utf-8

its = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
seeds = [30496, 260819, 240213, 10510, 150364]
dataset = "pubmed"

its.each do |it|
  seeds.each do |seed|
    puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv -i #{it} -s #{seed}"
    puts "../ktsne ../data/#{dataset}/#{dataset}_d_50.csv -i #{it} -k 5 -K 50 -s #{seed}"
  end
end
