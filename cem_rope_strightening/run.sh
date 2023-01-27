#!/bin/bash

#$((i = 0));
#i = 1;

#while (($i < 3));
#do
#echo "i=$i"; ((i=$i+1));
#done
time=$(date "+%Y%m%d%H%M%S")
echo $time

for((i=1;i<=1;i++));
do
echo "i = $i";
python run_cem_rope_flattening.py --test_num $i --log_dir "data/random/cem/$time/";
done

