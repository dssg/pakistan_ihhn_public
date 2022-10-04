#!/bin/bash
cd $( dirname "${BASH_SOURCE[0]}")
export PYTHONHASHSEED=1
psql --single-transaction -v ON_ERROR_STOP=1<<EOSQL
SET ROLE "pakistan-ihhn-role";
CREATE SCHEMA IF NOT EXISTS processed;
ALTER DEFAULT PRIVILEGES IN SCHEMA processed
GRANT SELECT ON TABLES to "pakistan-ihhn-role";

ALTER DEFAULT PRIVILEGES IN SCHEMA processed
GRANT USAGE ON SEQUENCES to "pakistan-ihhn-role";

ALTER DEFAULT PRIVILEGES IN SCHEMA processed
GRANT ALL ON SEQUENCES to "pakistan-ihhn-role";

ALTER DEFAULT PRIVILEGES IN SCHEMA processed
GRANT ALL ON FUNCTIONS to "pakistan-ihhn-role";
DROP TABLE IF EXISTS processed.codes,
    processed.transactions,
    processed.admissions CASCADE;
EOSQL
pakistan-ihhn process-codes
pakistan-ihhn process-transactions
pakistan-ihhn process-admissions

psql --single-transaction -v ON_ERROR_STOP=1<<EOSQL
REASSIGN OWNED BY ${PGUSER} TO "pakistan-ihhn-role";
EOSQL

psql --single-transaction -v ON_ERROR_STOP=1 -f processed.sql
