#!/bin/bash

datapath=$HOME/OPUS-MT-train/work/data/simple

if [ "X$1" = "X" ]; then
    echo "Give filename as argument"
    exit
fi

if [ "X$2" = "X" ]; then
    echo "Give filename as argument"
    exit
fi

if [ "X$3" = "X" ]; then
    echo "Give dataset name as argument"
    exit
fi

if [ ! -f $1 ]; then
    echo "No such file: $1"
    exit
fi

if [ ! -f $2 ]; then
    echo "No such file: $2"
    exit
fi

file1=$1
file2=$2
datasetname=$3

lang1=`echo $1 | cut -d . -f 2`
lang2=`echo $2 | cut -d . -f 2`

setfile1="$datasetname.${lang1}-${lang2}.clean.${lang1}.gz"
setfile2="$datasetname.${lang1}-${lang2}.clean.${lang2}.gz"

echo $setfile1
echo $setfile2

gzip -c $file1 > $datapath/$setfile1
gzip -c $file2 > $datapath/$setfile2
