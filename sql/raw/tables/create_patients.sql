CREATE SCHEMA IF NOT EXISTS raw;
---patients
CREATE TABLE IF NOT EXISTS raw.patients(
    new_mr TEXT,
    patient_type TEXT,
    patient_age NUMERIC,
    area_code TEXT,
    --using text rather than numeric because many area codes begin with zero,
    dist_code TEXT,
    --using text rather than numeric because many area codes begin with zero, 
    district TEXT,
    city_code TEXT --using text rather than numeric because many area codes begin with zero
);