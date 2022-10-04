DROP TABLE IF EXISTS
raw.nurses,
raw.procedures,
raw.admissions,
raw.supplies,
raw.investigations,
raw.decisions,
raw.doctors,
raw.transactions, 
raw.patients,
raw.codes,
raw.abbreviations,
raw.icd10cm_codes_2023,
raw.priority_codes,
raw.icd10cm_order_2023 CASCADE;

-- Tables for IHHN Data Model
SET ROLE "pakistan-ihhn-role";
\i tables/create_patients.sql
\i tables/create_providers.sql
\i tables/create_transactions.sql
\i tables/create_decisions.sql
\i tables/create_investigations.sql
\i tables/create_supplies.sql
\i tables/create_admissions.sql
\i tables/create_codes.sql
\i tables/create_procedures.sql;
\i tables/create_abbreviations.sql;
\i tables/create_official_codes.sql;
\i tables/create_priority_codes.sql;

