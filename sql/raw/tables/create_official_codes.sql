CREATE TABLE IF NOT EXISTS raw.icd10cm_codes_2023(icd_10_cm TEXT, description_long TEXT);
CREATE TABLE IF NOT EXISTS raw.icd10cm_order_2023(
    order_number TEXT,
    icd_10_cm TEXT,
    valid_for_submission BOOLEAN,
    description_short TEXT,
    description_long TEXT
);