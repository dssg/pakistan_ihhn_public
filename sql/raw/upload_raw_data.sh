#!/bin/bash
cd $( dirname "${BASH_SOURCE[0]}")

psql --single-transaction -v ON_ERROR_STOP=1<<EOSQL
SET ROLE "pakistan-ihhn-role";
CREATE SCHEMA IF NOT EXISTS raw;
ALTER DEFAULT PRIVILEGES IN SCHEMA raw
GRANT SELECT ON TABLES to "pakistan-ihhn-role";

ALTER DEFAULT PRIVILEGES IN SCHEMA raw
GRANT USAGE ON SEQUENCES to "pakistan-ihhn-role";

ALTER DEFAULT PRIVILEGES IN SCHEMA raw
GRANT ALL ON SEQUENCES to "pakistan-ihhn-role";

ALTER DEFAULT PRIVILEGES IN SCHEMA raw
GRANT ALL ON FUNCTIONS to "pakistan-ihhn-role";
EOSQL

set -e
processed_csv_dir="$1/processed_csvs"
psql --single-transaction -v ON_ERROR_STOP=1 -f raw.sql

# iterate through all files
for csvfile in ${processed_csv_dir}/*.csv; do
    echo $csvfile
    basefile="${csvfile##*/}"
    table="${basefile%%.*}"
    echo "Processing $table"
    psql --single-transaction -v ON_ERROR_STOP=1<<EOSQL
    \COPY raw.$table from '$csvfile' WITH CSV HEADER DELIMITER ',';
EOSQL
done