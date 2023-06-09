#!/bin/bash

user_agent='"nathante teblunthuis <nathante@uw.edu>"'
output_dir='/gscratch/comdata/raw_data/reddit_dumps/submissions'
base_url='https://files.pushshift.io/reddit/submissions/'

wget -r --no-parent -A 'RS_20*.bz2' --user-agent=$user_agent -P $output_dir -nd -nc $base_url
wget -r --no-parent -A 'RS_20*.xz' --user-agent=$user_agent -P $output_dir -nd -nc $base_url
wget -r --no-parent -A 'RS_20*.zst' --user-agent=$user_agent -P $output_dir -nd -nc $base_url
wget -r --no-parent -A 'RS_20*.bz2' --user-agent=$user_agent -P $output_dir -nd -nc $base_url/old_v1_data/
wget -r --no-parent -A 'RS_20*.xz' --user-agent=$user_agent -P $output_dir -nd -nc $base_url/old_v1_data/
wget -r --no-parent -A 'RS_20*.zst' --user-agent=$user_agent -P $output_dir -nd -nc $base_url/old_v1_data/

./check_submission_shas.py
