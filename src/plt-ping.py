import datetime
import time
import sys

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py

from my_db import MyDb
import utils
import colors

chart_name = 'ping'

host_info = {
    'www.ua': ['External', 4],
    '192.168.1.1': ['Router', 3],
    '192.168.1.64': ['old_hik', 2],
    '192.168.1.70': ['bullet', 1],
    '192.168.1.165': ['door_bell', 0],
}


# noinspection PyUnresolvedReferences
class PltPingNotUpdatable:
    """
    plot on plot.ly incrementally: init/update chart
    """
    df_hosts = None
    # @formatter:off
    grade_info = [
        {'level': 'disconnect', 'left': -np.inf, 'right': -1,     'color_scale': [0.0, 'red']},
        {'level': 'poor',       'left': 7,      'right': np.inf, 'color_scale': [0.3, 'orange']},
        {'level': 'moderate',   'left': 5,      'right': 7,      'color_scale': [0.6, 'rgb(235,206,135)']},
        {'level': 'good',       'left': -1,     'right': 5,      'color_scale': [1.0, 'green']}, #green
    ]
    # @formatter:on

    def __init__(self):
        self.db = MyDb()  # connection to MySQL

    def _get_ping_history(self) -> pd.core.frame.DataFrame:
        query = 'select * from ping order by time asc'
        df = pd.read_sql(query, self.db.mydb)
        return df

    @classmethod
    def _set_grade(cls, df: pd.core.frame.DataFrame):
        """ avg_rtt --> grade
        """
        df['grade'] = pd.Series(index=df.index)
        for gr in cls.grade_info:
            df.loc[(gr['left'] < df.avg_rtt) & (df.avg_rtt <= gr['right']), 'grade'] = gr['color_scale'][0]
        # df.loc[df.avg_rtt == -1, 'grade'] = 0.0  # no connect
        # df.loc[(-1 < df.avg_rtt) & (df.avg_rtt <= 5), 'grade'] = 1.0  # good
        # df.loc[(5 < df.avg_rtt) & (df.avg_rtt <= 8), 'grade'] = 0.6  # moderate
        # df.loc[8 < df.avg_rtt, 'grade'] = 0.3  # poor
        print('grade: ',df['grade'].value_counts())

    @classmethod
    def _prepare_data(cls, df: pd.core.frame.DataFrame):
        df.index = df['time']
        # del df['time']
        df.index = df.index.floor('1T')  # truncate to minutes

        cls._set_grade(df)

        # hosts --> columns
        df = df.groupby([df.index, 'host'])['grade'].agg('mean').unstack()  # minutes * hosts [avg_rtt]  avg_rtt
        globals()['dfm']=df.copy()

        # df = df.resample('1H').mean()
        # globals()['dfh']=df.copy()

        # sort columns by seqn from host_info
        df.rename(lambda s: f'{host_info[s][1]:02d}.{s}', axis='columns', inplace=True)  # host -> <seqn>.host
        df.sort_index(axis=1, inplace=True)  # sort by seqn
        df.rename(lambda s: host_info[s[3:]][0], axis='columns', inplace=True)  # <seqn>.host -> host_name

        # dict_cat={}
        # for c in df.columns:
        #     s = pd.cut(pd.Series(df[c],index=df.index), bins=[-np.inf, -1.0, 4.0, 8.0, np.inf])
        #     s.cat.categories = ['disconnect', 'good', 'moderate', 'poor']
        #     s.cat.reorder_categories(['good','moderate','poor','disconnect'])
        #     dict_cat[c] = s
        # df = pd.DataFrame(data=dict_cat,index=df.index)
        return df  # df

    @classmethod
    def _create_fig(cls, df):
        print('grade info: ',[ gr['color_scale'] for gr in cls.grade_info])
        data = [
            go.Heatmap(
                z=df.T.values.tolist(),
                x=df.index.tolist(),
                y=df.columns,
                colorscale=[ gr['color_scale'] for gr in cls.grade_info],
                ygap=4,
                xgap=1
            )
        ]
        layout = go.Layout(
            title='ping stat',
            # xaxis=dict(ticks=''),  # nticks=12),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label='1 day',
                             step='day',
                             stepmode='backward'),
                        dict(count=1,
                             label='1 hour',
                             step='hour',
                             stepmode='backward'),
                        dict(count=1,
                             label='1 minute',
                             step='minute',
                             stepmode='backward'),
                        dict(step='all')
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type='date'
            ),
            yaxis=dict(ticks='')
        )
        fig = go.Figure(data=data, layout=layout)
        return fig

    def draw(self):
        """ overwrite chart based on history
        """
        df = self._get_ping_history()
        df_p = self._prepare_data(df)
        fig = self._create_fig(df_p)
        r = py.iplot(fig, filename=chart_name, fileopt='overwrite')
        print(f'plot created at {r.resource}.')


# noinspection PyUnresolvedReferences
class PltPing(PltPingNotUpdatable):
    def __init__(self):
        super().__init__()
        self._last_timestamp: datetime.datetime = None  # last timestamp been sent to chart server

    def _update_last_timestamp(self, df: pd.core.frame.DataFrame):
        t = df[-1:]['time']
        self._last_timestamp = t.dt.to_pydatetime()[0]
        print(f'last timestamp = {self._last_timestamp}')

    def _get_ping_history(self) -> pd.core.frame.DataFrame:
        df = super()._get_ping_history()
        self._update_last_timestamp(df)
        return df

    def _get_ping_newdata(self) -> pd.core.frame.DataFrame:
        """ get new records from ping """
        last_time = MyDb.datetime_2_sql(self._last_timestamp)
        query = f"""select * from ping where time > str_to_date('{last_time}','%Y-%m-%d %T') order by time asc"""
        # print('query: ', query)
        df = pd.read_sql(query, self.db.mydb)
        self._update_last_timestamp(df)
        return df

    def start(self):
        """ overwrite chart based on all previous history """
        super().draw()

    def update(self):
        """ update chart based on last data (which have been inserted after last update)
        """
        df = self._get_ping_newdata()
        df = self._prepare_data(df)
        print('update:  df.shape=', df.shape)
        fig = self._create_fig(df)
        r = py.iplot(fig, filename=chart_name, fileopt='extend')
        print(f'       =========== {df.shape[0]} timemarks added to {r.resource}')


if __name__ == '__main__':
    # plt = PltPing()
    plt = PltPing()
    plt.start()

    iter_numb = 0
    for iter_ in range(iter_numb):
        print(f'iter={iter_}/{iter_numb}')
        time.sleep(100)
        plt.update()
