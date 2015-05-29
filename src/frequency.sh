#!/bin/bash

data_dir=${1:-.}
workspace_dir=${2:-.}

echo "data path:" $data_dir
echo "workspace path:" $workspace_dir

output_dir=$workspace_dir/frequency
mkdir -p $output_dir
for column in "auction" "bidder_id" "merchandise" "device" "country" "ip" "url"
do
    sqlite3 "$data_dir/bids.db" -cmd ".header off" -cmd ".mode csv" \
            "SELECT $column, COUNT(*) AS n FROM bids GROUP BY $column ORDER BY n;" \
            > "$output_dir/$column.csv"
done
