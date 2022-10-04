#!/bin/sh

python -m pip install --editable .

# Enter data directory
excel_dir=$1
if [ ! -d "$1/raw_csvs" ]; then
  mkdir "$1/raw_csvs"
fi
raw_csv_dir="$1/raw_csvs"

if [ ! -d "$1/processed_csvs" ]; then
  mkdir "$1/processed_csvs"
fi

processed_csv_dir="$1/processed_csvs"

find . -maxdepth 1 -type f

## READING XLSX AND XLS TO CSV
read_xls_to_csv () {
  # Read HMIS xls files
  for file in find "$excel_dir"/* -maxdepth 1 -type f;
    do
      echo $file
      basefile="${file##*/}"
      csv_file="${raw_csv_dir}/${basefile%%.*}.csv"
      if [[ $file = *".xls" ]]; then
        echo "Reading xls data to csv"
        in2csv $file --format xls --write-sheets "-" > $csv_file;
      fi
  done
  pakistan-ihhn process-xlsx $excel_dir $raw_csv_dir
  pakistan-ihhn process-official-codes $excel_dir $raw_csv_dir
  pakistan-ihhn process-csv $raw_csv_dir $processed_csv_dir
}

# Call function
read_xls_to_csv

# creating raw schema
bash sql/raw/upload_raw_data.sh $1

# creating processed schema
bash sql/processed/upload_processed_data.sh 

# creating modeling schema
bash sql/modeling/upload_model_prep_schema.sh

# run machine learning pipeline 
pakistan-ihhn run-pipeline --schema_type dev