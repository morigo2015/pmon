-- MySQL Script generated by MySQL Workbench
-- нд, 11-лис-2018 11:49:29 +0200
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
SHOW WARNINGS;
-- -----------------------------------------------------
-- Schema mysql
-- -----------------------------------------------------
SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `polygons`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `polygons` (
  `fname` VARCHAR(80) NOT NULL,
  `points_lst` VARCHAR(80) NULL DEFAULT NULL,
  `left_angle` INT NULL DEFAULT NULL,
  `right_angle` INT NULL DEFAULT NULL,
  PRIMARY KEY (`fname`));

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `transforms`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `transforms` (
  `fname` VARCHAR(80) NOT NULL,
  `mat` VARCHAR(80) NULL DEFAULT NULL,
  PRIMARY KEY (`fname`));

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `combinations`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `combinations` (
  `fname_pic` VARCHAR(80) NULL,
  `fname_trans` VARCHAR(80) NULL,
  `trans_points_lst` VARCHAR(80) NULL DEFAULT NULL,
  `trans_left_ang` INT NULL DEFAULT NULL,
  `trans_right_ang` INT NULL DEFAULT NULL);

SHOW WARNINGS;

-- -----------------------------------------------------
-- View `comb_detailed`
-- -----------------------------------------------------
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
        c.fname_trans = t.fname;
SHOW WARNINGS;

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
