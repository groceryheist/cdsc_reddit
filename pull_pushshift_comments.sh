#!/bin/bash

user_agent='nathante teblunthuis <nathante@uw.edu>'
output_dir='/gscratch/comdata/raw_data/reddit_dumps/comments'
base_url='https://files.pushshift.io/reddit/comments/'

wget -r --no-parent -A 'RC_20*.bz2' -U $user_agent -P $output_dir -nd -nc $base_url
wget -r --no-parent -A 'RC_20*.xz' -U $user_agent -P $output_dir -nd -nc $base_url
wget -r --no-parent -A 'RC_20*.zst' -U $user_agent -P $output_dir -nd -nc $base_url



