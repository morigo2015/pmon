use mysql;
-- create table twostr (str1 varchar(80), str2 varchar(80) );
-- create table poly (fname varchar(80), x1 int, y1 int, x2 int, y2 int, x3 int, y3 int, x4 int, y4 int);
show global variables like 'local_infile';
set global local_infile='ON';
load data local infile '/home/im/mypy/vint/src/tst.txt' into table poly
fields terminated by ',' enclosed by '"' lines terminated by '\n';

create table if not exists polygons (fname varchar(80), points_lst varchar(80));
select * from polygons;
drop table polygons;
delete from polygons where fname='test_db';

select * from polygons where fname like 'test%';

select 
	min(left_angle) as min_l, max(left_angle) as max_l, 
	min(right_angle) as min_r, max(right_angle) as max_r,
    min(right_angle-left_angle) as min_delt, max(right_angle-left_angle) as max_delt,
    min((right_angle+left_angle)/2) as min_avg, max((right_angle+left_angle)/2) as max_avg
from polygons;

select fname, (right_angle+left_angle)/2 as avg from polygons;
describe polygons;

create table poly_save as select * from polygons;
select * from poly_save;

create or replace view pview as 
	select fname, left_angle, right_angle, 
		(right_angle-left_angle) as delta_angle,
        (right_angle+left_angle)/2 as cent_angle
    from polygons
    where fname != 'clb7'
    ;
select * from pview;
select * from polygons as p1,poly_save as p2 where p1.fname=p2.fname; 
select * from comb_detailed;
show warnings;

insert into transforms values ("aa","bb",1,2);
drop table combinations;
select * from transforms;
delete from transforms where fname = "aa";
SELECT * FROM combinations;
select * from comb_detailed;

create or replace view angle_min_max as 
select trans, min(trans_centr_ang) as min_ang, max(trans_centr_ang) as max_ang, 
from comb_detailed
GROUP BY trans;
