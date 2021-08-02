#!/usr/bin/env ruby
# encoding: utf-8

ps = [5, 25, 50]
ks = [5, 50, 100]

ps.each do |p|
  ks.each do |k|
    puts "../ktsne -p #{p} -k #{k} -K #{k} ../data/color_n_10000.csv"
  end
end
