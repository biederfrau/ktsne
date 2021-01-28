#!/usr/bin/env ruby
# encoding: utf-8

require 'fileutils'

csv_files = ARGV.sort_by { |f| f[/.*it_(\d+).csv/, 1].to_i }
csv_files.each { |csv| `Rscript plot_embedding.R #{csv}` }

jpg_files = csv_files.map { |f| f.sub(".csv", ".jpg") }
gif_name = csv_files[0].sub(/_it_\d+\.csv/, ".gif")

`convert -delay 0 -loop 0 #{jpg_files.join " "} #{gif_name}`

exit if $? != 0

FileUtils.rm csv_files
FileUtils.rm jpg_files
