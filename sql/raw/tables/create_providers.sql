---doctors
CREATE TABLE IF NOT EXISTS raw.doctors(
    new_mr TEXT,
    doctor_code TEXT,
    doctor_id TEXT,
    specialty TEXT
);
CREATE TABLE IF NOT EXISTS raw.nurses(nurse_id TEXT);