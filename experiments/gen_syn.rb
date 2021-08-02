#!/usr/bin/env ruby
# encoding: utf-8

# ns = [100, 500, 1000, 5000, 10000, 50000, 100000]#, 500000, 1000000]
ns = [500000, 1000000]
seeds = [30496, 260819, 240213, 10510, 150364]
d = 50

ns.each do |n|
  seeds.each do |seed|
    puts "../ktsne ../data/syn2/syn2_n_#{n}.csv -s #{seed}"
    puts "python3 ../bhtsne/bhtsne.py -i ../data/syn2/syn2_n_#{n}.csv -o bhtsne_syn2_n_#{n}_s_#{seed}.tsv --randseed #{seed} --no_pca -p 30"
    # puts "python3 ../scripts/run_umap.py ../data/syn2/syn2_n_#{n}.csv #{seed}"
    # puts "python3 ../scripts/run_fitsne.py ../data/syn2/syn2_n_#{n}.csv #{seed}"
  end
end
