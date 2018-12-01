use mysql;
-- create table twostr (str1 varchar(80), str2 varchar(80) );
-- create table poly (fname varchar(80), x1 int, y1 int, x2 int, y2 int, x3 int, y3 int, x4 int, y4 int);

show global variables like 'local_infile';
set global local_infile='ON';

delete from poly;

load data local infile '/home/im/mypy/vint/src/tst.txt' into table poly
fields terminated by ',' enclosed by '"' lines terminated by '\n';

select * from poly;


