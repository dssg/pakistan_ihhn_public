psql --single-transaction -v ON_ERROR_STOP=1<<EOSQL
REASSIGN OWNED BY ${PGUSER} TO "pakistan-ihhn-role";
EOSQL

#!/bin/bash
cd $( dirname "${BASH_SOURCE[0]}")
export PYTHONHASHSEED=1
psql --single-transaction -v ON_ERROR_STOP=1<<EOSQL
SET ROLE "pakistan-ihhn-role";
CREATE SCHEMA IF NOT EXISTS model_output;
ALTER DEFAULT PRIVILEGES IN SCHEMA model_output
GRANT SELECT ON TABLES to "pakistan-ihhn-role";

ALTER DEFAULT PRIVILEGES IN SCHEMA model_output
GRANT USAGE ON SEQUENCES to "pakistan-ihhn-role";

ALTER DEFAULT PRIVILEGES IN SCHEMA model_output
GRANT ALL ON SEQUENCES to "pakistan-ihhn-role";

ALTER DEFAULT PRIVILEGES IN SCHEMA model_output
GRANT ALL ON FUNCTIONS to "pakistan-ihhn-role";
DROP TABLE IF EXISTS model_output.train,
    model_output.retraining CASCADE;
EOSQL

psql --single-transaction -v ON_ERROR_STOP=1 -f tables/create_subset_model_prep_train.sql
psql --single-transaction -v ON_ERROR_STOP=1 -f tables/create_retraining.sql

psql --single-transaction -v ON_ERROR_STOP=1<<EOSQL
REASSIGN OWNED BY ${PGUSER} TO "pakistan-ihhn-role";
EOSQL
