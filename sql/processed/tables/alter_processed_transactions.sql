---transactions
ALTER TABLE processed.transactions
ADD PRIMARY KEY (new_mr, new_er, ed_dx_hash);
-- ALTER TABLE processed.transactions
-- ADD CONSTRAINT ed_dx_fk FOREIGN KEY (ed_dx_hash) REFERENCES processed.codes (ed_dx_hash);