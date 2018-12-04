import json
import datetime

import cv2
import numpy as np
import mysql.connector


class MyDb:

    def __init__(self, db_param={'host': 'localhost', 'user': 'root', 'passwd': '121212', 'database': 'mysql'}):
        self.db_param = db_param
        self.mydb = mysql.connector.connect(
            host=db_param['host'], user=db_param['user'], passwd=db_param['passwd'], database=db_param['database']
            , autocommit=True
            , charset='latin1'
            # ,use_unicode=True
            # ,buffered=True
        )
        self.curs = self.mydb.cursor()
        print(f"connected to {db_param['user']}:{db_param['database']} host={db_param['host']}")

    def __del__(self):
        # self.curs.close()
        self.mydb.close()

    def exec(self, statement, args=None, show=True):
        if show:
            print(f"Exec:{statement}")
        try:
            self.curs.execute(statement, args)
        except  mysql.connector.Error as err:
            print(f"Something went wrong while exec {statement}: {format(err)}")
            raise
        # if len(self.curs):
        result_str = [s for s in self.curs]
        if show:
            print("result:",end='')
            for s in result_str:
                print(s)
            print("\n")
        return result_str

    """
    auxiliary functions for MySQL
    """

    @staticmethod
    def datetime_2_sql(dt: datetime.datetime) -> str:
        """ python datetime.datetime --> MySQL datetime """
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def sql_2_datetime(s: str) -> datetime.datetime:
        """ sql datetime --> python datetime.datetime """
        return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

    @classmethod
    def now_timestamp(cls) -> str:
        """ python datetime.datetime.now() --> sql datetime """
        return cls.datetime_2_sql(datetime.datetime.now())

    @staticmethod
    def bool_2_sql(val:bool) -> str:
        """ python bool -> sql bool """
        return str(val).upper()

if __name__ == '__main__':
    img_fname = '/home/im/mypy/Homography/book.jpg'
    img = cv2.imread(img_fname)
    db = MyDb()
    tbl_name = 'images'
    # db.exec('select * from poly;')
    # db.exec(f'create table if not exists {tbl_name} (file_id varchar(80), img blob);')
    # db.exec('show tables;')
    db.exec('insert into transforms values ("aa","bb",1,2)')

    # db.exec(f'select * from {tbl_name};')
    # db.save_img_to_db(img_fname,img,tbl_name)
    # img2=db.load_img_from_db(img_fname)
    # img_fname_splitted = img_fname.split(sep='.')
    # new_fname = img_fname_splitted[-2]+"-loaded"+img_fname_splitted[-1]
    # cv2.imwrite(new_fname,img2)
