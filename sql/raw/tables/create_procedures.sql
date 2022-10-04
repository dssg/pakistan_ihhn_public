---procedures
CREATE TABLE IF NOT EXISTS raw.procedures(
    new_mr TEXT,
    doctor_code TEXT,
    procedure_sequence NUMERIC,
    procedure_code NUMERIC,
    procedure TEXT,
    proceduredate TIMESTAMP,
    proc_doctor_code NUMERIC
);