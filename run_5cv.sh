#!/bin/bash

for i in {1..5}
do
 ver="v"$i
 output="log_5cv_three_m50_ft_no_fix_v"$i".txt"
 echo $output
 python3 run_5cv_adc_scc_lcc_add_lung3_m50_ft_no_fix.py $ver > $output
done

