#!/usr/bin/env ruby
# encoding: utf-8

require 'fileutils'

exit -1 if File.exist? 'pid'
File.write 'pid', Process.pid

at_exit { File.unlink 'pid' }

last_mtime = Time.new 0
loop do
  sleep 1
  mtime = Dir['**/*.{tex,bib}'].map { |f| File.mtime f }.max
  next if mtime.nil? || mtime <= last_mtime

  `rubber --pdf slides 2> /dev/null`
  `rubber --clean slides`
  `mv slides.pdf view.pdf`
  last_mtime = mtime
end
