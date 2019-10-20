#!/usr/bin/env bash

convert \
    -delay 0 \
    $(for i in $(seq 0 1 199); do echo "img/$1-${i}.png"; done) \
    -loop 0 \
    $2
