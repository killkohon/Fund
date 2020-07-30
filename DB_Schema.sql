PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE fund_values (
   vid  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
   fundcode  VARCHAR(255) NOT NULL,
   netvalue  DECIMAL(10,5) NOT NULL,
   unfixedvalue  DECIMAL(10,5) NOT NULL,
   vdate  DATE NOT NULL,
   buyable  BOOLEAN NOT NULL DEFAULT '1',
   salable  BOOLEAN NOT NULL DEFAULT '1',
   dividend  VARCHAR(100) DEFAULT NULL
);
CREATE TABLE fund_meta (
   fundcode  VARCHAR(20) NOT NULL  PRIMARY KEY,
   fundname  VARCHAR(200) NOT NULL,
   memo  VARCHAR(2000) DEFAULT NULL,
   category  VARCHAR(20) DEFAULT NULL,
   categoryname  VARCHAR(100) DEFAULT NULL,
   company  VARCHAR(100) DEFAULT NULL,
   manager  VARCHAR(100) DEFAULT NULL
);
DELETE FROM sqlite_sequence;
CREATE UNIQUE INDEX fundcode_vdate_idx on fund_values (fundcode,vdate);
CREATE UNIQUE INDEX fundmeta_code_idx on fund_meta (fundcode);
COMMIT;