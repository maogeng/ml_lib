#!/bin/bash

## Usage ################################
# ./nb2script <file-name without extension>
# Example:
# nb2script binary_lr
#########################################

if [ $# -ne "1" ]; then
	echo "Usage: ./nb2script <file-name without extension>"
else
	jupyter nbconvert --to script $1.ipynb > $1.py
fi
