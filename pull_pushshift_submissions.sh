#!/bin/bash

user_agent='nathante teblunthuis <nathante@uw.edu>'
output_dir='/gscratch/comdata/raw_data/reddit_dumps/submissions'
base_url='https://files.pushshift.io/reddit/submissions/'

wget -r --no-parent -A 'RS_20*.bz2' -U $user_agent -P $output_dir -nd -nc $base_url
wget -r --no-parent -A 'RS_20*.xz' -U $user_agent -P $output_dir -nd -nc $base_url
wget -r --no-parent -A 'RS_20*.zst' -U $user_agent -P $output_dir -nd -nc $base_url
wget -r --no-parent -A 'RS_20*.bz2' -U $user_agent -P $output_dir -nd -nc $base_url/old_v1_data/
wget -r --no-parent -A 'RS_20*.xz' -U $user_agent -P $output_dir -nd -nc $base_url/old_v1_data/
wget -r --no-parent -A 'RS_20*.zst' -U $user_agent -P $output_dir -nd -nc $base_url/old_v1_data/

./check_submissions_shas.py
