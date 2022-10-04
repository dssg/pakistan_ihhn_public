CREATE TABLE IF NOT EXISTS model_output.train(
    new_er TEXT,
    new_mr TEXT,
    gender TEXT,
    city TEXT,
    age_years NUMERIC,
    triage_datetime TIMESTAMP,
    triagecomplaint TEXT,
    bp TEXT,
    tr_pulse NUMERIC,
    tr_temp NUMERIC,
    tr_resp NUMERIC,
    acuity TEXT,
    visit_datetime TIMESTAMP,
    hopi TEXT,
    disposition TEXT,
    disposition_time TIMESTAMP,
    ed_dx_hash BIGINT,
    ed_dx TEXT,
    category TEXT,
    code TEXT
);
INSERT INTO model_output.train
SELECT b.new_er,
    b.new_mr,
    b.gender,
    b.city,
    b.age_years,
    b.triage_datetime,
    b.triagecomplaint,
    b.bp,
    b.tr_pulse,
    b.tr_temp,
    b.tr_resp,
    b.acuity,
    b.visit_datetime,
    b.hopi,
    b.disposition,
    b.disposition_time,
    a.ed_dx_hash,
    a.ed_dx,
    a.category,
    a.code
FROM processed.codes a
    INNER JOIN processed.transactions b on btrim(lower(a.ed_dx)) = btrim(lower(b.ed_dx))
    and a.age_years = b.age_years
    and btrim(lower(a.hopi)) = btrim(lower(b.hopi))
    and btrim(lower(a.triagecomplaint)) = btrim(lower(b.triagecomplaint));
ALTER TABLE model_output.train
ADD COLUMN unique_id SERIAL;