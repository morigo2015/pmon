
USE mysql;

-- Table with data (filled by pythons scripts):
CREATE TABLE IF NOT EXISTS polygons (
    fname VARCHAR(80),
    points_lst VARCHAR(250),
    left_angle INT,
    right_angle INT
	);

CREATE TABLE IF NOT EXISTS transforms (
    fname VARCHAR(80),
    mat VARCHAR(250),
    width INT,
    height INT
	);

CREATE TABLE IF NOT EXISTS combinations (
    fname_pic VARCHAR(80),
    fname_trans VARCHAR(80),
    trans_points_lst VARCHAR(250),
    trans_left_ang INT,
    trans_right_ang INT
);
    
-- Views based on data tables:
CREATE OR REPLACE VIEW comb_detailed AS
    SELECT 
        c.fname_pic AS orig,
        p.left_angle AS orig_left_ang,
        p.right_angle AS orig_right_ang,
        (p.left_angle + p.right_angle) / 2 AS orig_centr_ang,
        c.fname_trans AS trans,
        c.trans_left_ang AS trans_left_ang,
        c.trans_right_ang AS trans_right_ang,
        (c.trans_left_ang + c.trans_right_ang) / 2 AS trans_centr_ang,
        p.points_lst AS orig_points_lst,
        c.trans_points_lst AS trans_points_lst,
        t.mat AS mat
    FROM
        combinations AS c,
        polygons AS p,
        transforms AS t
    WHERE
        c.fname_pic = p.fname
            AND c.fname_trans = t.fname
	;

CREATE OR REPLACE VIEW trans_min_max AS 
	SELECT trans, MIN(trans_centr_ang) AS min_ang, MAX(trans_centr_ang) AS max_ang
	FROM comb_detailed
	GROUP BY trans;

CREATE OR REPLACE VIEW trans_delta AS
	SELECT *, (max_ang-min_ang) AS delta 
    FROM trans_min_max
    ORDER BY delta;
