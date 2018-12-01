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
        c.fname_pic   = p.fname AND 
        c.fname_trans = t.fname