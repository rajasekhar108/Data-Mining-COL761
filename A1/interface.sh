#!/bin/bash
if [ "$1" == "C" ]; then
        ./fpattern "$2" 
        ./compress frq.txt "$2" "$3"
fi
if [ "$1" == "D" ]; then
        ./decrypt "$2" "$3"
fi