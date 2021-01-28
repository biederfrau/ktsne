#!/usr/bin/env ruby
# encoding: utf-8

require 'benchmark'

results = {}
puts "n,t"
ARGV.sort_by { |f| f[/.*n_(\d+).*/, 1].to_i }.each do |fname|
  n = fname[/.*n_(\d+).*/, 1].to_i

  t = Benchmark.realtime { `../ktsne #{fname}` }
  puts "#{n},#{t}"
end
