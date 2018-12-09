

import cv2
import numpy as np
import json
import mysql.connector

class MyDb:

    def __init__(self,db_param = {'host':'localhost','user':'root', 'passwd':'121212', 'database':'mysql'}):
        self.db_param=db_param
        self.mydb=mysql.connector.connect(
            host=db_param['host'],user=db_param['user'],passwd=db_param['passwd'],database=db_param['database']
            ,autocommit=True
            ,charset='latin1'
            # ,use_unicode=True
            #,buffered=True
            )
        self.curs = self.mydb.cursor()
        print(f"connected to {db_param['user']}:{db_param['database']} host={db_param['host']}")

    def __del__(self):
        # self.curs.close()
        self.mydb.close()

    def exec(self,statement,args=None):
        print(f"\nExec:{statement}:\n")
        try:
            self.curs.execute(statement,args)
        except  mysql.connector.Error as err:
            print(f"Something went wrong while exec {statement}: {format(err)}")
            raise
        #if len(self.curs):
        result_str = [s for s in self.curs]
        for s in result_str:
            print(s)
        return result_str


    def get_fname_lst(self,fname_pattern,table_name='polygons'):
        fname_tuple_lst = self.exec(f'select fname from {table_name} where fname like "{fname_pattern}"')
        return [ fn[0] for fn in fname_tuple_lst]

    def save_points_to_db(self,img_fname,
                          points_list, side_angles,
                          delete_old=False, table_name='polygons'):
        self.exec(
            f'create table if not exists {table_name} '
            f'    (fname varchar(80), points_lst varchar(80), left_angle int, right_angle int)'
            )
        # self.exec(f'describe {table_name}')
        if delete_old:
            self.exec(f'delete from {table_name} where fname="{img_fname}"')
        json_points=json.dumps(points_list)
        self.exec(
            f'insert into {table_name} '
            f'    values ( "{img_fname}", "{json_points}", {side_angles[0]}, {side_angles[1]} )')

    def load_points_from_db(self,fname,table_name='polygons'):
        result_str = self.exec(
            f'select * from {table_name} '
            f'    where fname="{fname}"'
            )[0]
        json_str = result_str[1]
        list_lists = json.loads(json_str)
        points_list = [ tuple(i) for i in list_lists ]
        return points_list

    def save_matrix(self,fname,
                    matrix,width,height,
                    delete_old=True,table_name='transforms'):
        if delete_old:
            self.exec(f'delete from {table_name} where fname="{fname}"')
        json_mat=json.dumps(matrix.tolist())
        self.exec(
            f'insert into {table_name} '
            f'    values ( "{fname}", "{json_mat}", {width}, {height} )')

    def load_matrix(self,fname,table_name='transforms'):
        result_str = self.exec(
            f'select * from {table_name} '
            f'    where fname="{fname}"'
            )[0]
        lst = json.loads(result_str[1])
        return np.array(lst),result_str[2],result_str[3]  # (points_lst,width,height)

    def save_combination(self,fname_trans,fname_pic,
                         trans_points_lst,trans_left_ang,trans_right_ang,
                         delete_old=True,table_name='combinations'):
        if delete_old:
            self.exec(
                f'delete from {table_name} '
                f'where fname_pic="{fname_pic}" and fname_trans="{fname_trans}"')
        json_lst=json.dumps(trans_points_lst.tolist())
        self.exec(
            f'insert into {table_name} '
            f'    values ( "{fname_pic}", "{fname_trans}", "{json_lst}", '
            f'              {trans_left_ang}, {trans_right_ang} )')

    def save_img(self,img,id,table='frames'):
        self.curs.execute(f'INSERT INTO {table} VALUES ({id}, %s)', (img.tobytes(),))

    def load_img(self,id,dtype='uint8',shape=(720,1280,3),table='frames'):
        self.curs.execute(f'select * from {table} where id={id}')
        rest_s = self.curs.fetchall()[0][1]
        rest_img = np.frombuffer(np.array(bytes(rest_s, encoding='latin1')), dtype=dtype).reshape(shape)
        return rest_img


if __name__=='__main__':
    img_fname = '/home/im/mypy/Homography/book.jpg'
    img = cv2.imread(img_fname)
    db=MyDb()
    tbl_name = 'images'
    # db.exec('select * from poly;')
    #db.exec(f'create table if not exists {tbl_name} (file_id varchar(80), img blob);')
    #db.exec('show tables;')
    db.exec('insert into transforms values ("aa","bb",1,2)')

    # db.exec(f'select * from {tbl_name};')
    # db.save_img_to_db(img_fname,img,tbl_name)
    # img2=db.load_img_from_db(img_fname)
    # img_fname_splitted = img_fname.split(sep='.')
    # new_fname = img_fname_splitted[-2]+"-loaded"+img_fname_splitted[-1]
    # cv2.imwrite(new_fname,img2)

