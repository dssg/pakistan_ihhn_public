---admissions
ALTER TABLE processed.admissions
ADD PRIMARY KEY (new_mr, admission_no, diagnosis_hash);