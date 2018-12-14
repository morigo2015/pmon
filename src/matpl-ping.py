import datetime
import time
import sys
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

from matplotlib import cm

import numpy as np
import pandas as pd
# import plotly.graph_objs as go
# import plotly.plotly as py

from my_db import MyDb

chart_name = 'ping'

# @formatter:off
host_info = {
    'www.ua':        ['External',  4],
    '192.168.1.1':   ['Router',    3],
    '192.168.1.64':  ['old_hik',   2],
    '192.168.1.70':  ['bullet',    1],
    '192.168.1.165': ['door_bell', 0],
}
# @formatter:on

PandasFrame = pd.core.frame.DataFrame  # shortcut for annotations


# noinspection PyUnresolvedReferences
class _PltPingNotUpdatable:
    """
    plot on plot.ly incrementally: init/update chart
    """
    # @formatter:off
    grade_info = [
        {'level': 'disconnect', 'left': -np.inf, 'right': -1,     'color_scale': [0.0, 'red']},
        {'level': 'poor',       'left': 7,       'right': np.inf, 'color_scale': [0.3, 'orange']},
        {'level': 'moderate',   'left': 5,       'right': 7,      'color_scale': [0.6, 'rgb(235,206,135)']},
        {'level': 'good',       'left': -1,      'right': 5,      'color_scale': [1.0, 'green']},
    ]
    # @formatter:on

    def __init__(self):
        self.db = MyDb()  # connection to MySQL

    def _get_ping_history(self) -> PandasFrame:
        query = 'select * from ping order by time asc'
        df = pd.read_sql(query, self.db.mydb)
        return df

    @classmethod
    def _set_grade(cls, df: PandasFrame):
        """ avg_rtt --> grade
        """
        df['grade'] = pd.Series(index=df.index)
        for gr in cls.grade_info:
            df.loc[(gr['left'] < df.avg_rtt) & (df.avg_rtt <= gr['right']), 'grade'] = gr['color_scale'][0]
        print('grade: ', df['grade'].value_counts())

    # @classmethod
    # def _prepare_data_heatmap(cls, df: PandasFrame) -> PandasFrame:
    #     df.index = df['time']
    #     # del df['time']
    #     df.index = df.index.floor('1T')  # truncate to minutes
    #
    #     # avg_rtt --> grades
    #     cls._set_grade(df)
    #
    #     # hosts --> columns
    #     df = df.groupby([df.index, 'host'])['grade'].agg('mean').unstack()  # minutes * hosts [avg_rtt]  avg_rtt
    #     globals()['dfm'] = df.copy()
    #
    #     # df = df.resample('1H').mean()
    #     # globals()['dfh']=df.copy()
    #
    #     # sort columns by seqn from host_info
    #     df.rename(lambda s: f'{host_info[s][1]:02d}.{s}', axis='columns', inplace=True)  # host -> <seqn>.host
    #     df.sort_index(axis=1, inplace=True)  # sort by seqn
    #     df.rename(lambda s: host_info[s[3:]][0], axis='columns', inplace=True)  # <seqn>.host -> host_name
    #     return df
    #
    # @classmethod
    # def _create_heatmap(cls, df):
    #     print('grade info: ', [gr['color_scale'] for gr in cls.grade_info])
    #     data = [
    #         go.Heatmap(
    #             z=df.T.values.tolist(),
    #             x=df.index.tolist(),
    #             y=df.columns,
    #             colorscale=[gr['color_scale'] for gr in cls.grade_info],
    #             ygap=4,
    #             xgap=1
    #         )
    #     ]
    #     layout = go.Layout(
    #         title='ping stat',
    #         # xaxis=dict(ticks=''),  # nticks=12),
    #         xaxis=dict(
    #             rangeselector=dict(
    #                 buttons=list([
    #                     dict(count=1, label='1 day', step='day', stepmode='backward'),
    #                     dict(count=1, label='1 hour', step='hour', stepmode='backward'),
    #                     dict(count=1, label='1 minute', step='minute', stepmode='backward'),
    #                     dict(step='all')
    #                 ])
    #             ),
    #             rangeslider=dict(visible=True),
    #             type='date'
    #         ),
    #         yaxis=dict(ticks='')
    #     )
    #     fig = go.Figure(data=data, layout=layout)
    #     return fig
    #
    # def draw_heat(self):
    #     """ overwrite chart based on history
    #     """
    #     df = self._get_ping_history()
    #     df_p = self._prepare_data_heatmap(df)
    #     fig = self._create_heatmap(df_p)
    #     r = py.iplot(fig, filename=chart_name, fileopt='overwrite')
    #     print(f'plot created at {r.resource}.')

    @classmethod
    def _prepare_data(cls, df:PandasFrame):
        df.index = df['time']
        df.index = df.index.floor('1T')  # truncate to minutes

        df.avg_rtt.clip_upper(15.0,inplace=True)
        df.loc[df.avg_rtt==-1,'avg_rtt']=np.nan

        return df

    def draw_lines(self):
        df = self._get_ping_history()
        df = self._prepare_data(df)

        s = df.loc[df.host == 'www.ua', 'avg_rtt']
        globals()['s']=s
        s = s.tail(1000)
        s.plot()
        plt.show()

    def draw_heat(self):
        df = self._get_ping_history()
        df = self._prepare_data(df)

        # avg_rtt --> grades
        # self._set_grade(df)

        # hosts --> columns
        df = df.groupby([df.index, 'host'])['avg_rtt'].agg('mean').unstack()  # minutes * hosts [avg_rtt]  avg_rtt
        globals()['dfm'] = df.copy()

        df = df.resample('1H').mean()
        globals()['dfh']=df.copy()

        df=df.tail(24)
        fig,ax = plt.subplots()
        # cmap=cm.get_cmap('viridis',128)
        # cmap.set_under('black')
        print(np.nanmin(df.T),np.nanmax(df.T))
        heatmap = ax.pcolor(df.T, # cmap=cmap,  #plt.cm.viridis,
                   vmin=np.nanmin(df.T),vmax=np.nanmax(df.T),
                   edgecolors='k',linewidth=1) #   cm.get_cmap('viridis', 256))
        ax.patch.set(color='red') # hatch='x',edgecolor='red',fill=True,
        # heatmap.cmap.set_under('red')
        bar = fig.colorbar(heatmap) # , extend='both')
        plt.xticks(np.arange(0.5, len(df.index), 1), df.index, rotation='vertical') #,verticalalignment='bottom')
        plt.yticks(np.arange(0.5, len(df.columns), 1), df.columns)
        plt.savefig(fname='/home/im/mypy/pmon/tst/matpl-ping.png')
        plt.show()



# noinspection PyUnresolvedReferences
class PltPing(_PltPingNotUpdatable):
    """ Plot charts, based on MySQL table ping. Allows to update chart based on ping table updates, too.
    """

    def __init__(self):
        super().__init__()
        self._last_timestamp: datetime.datetime = None  # last timestamp been sent to chart server

    def _update_last_timestamp(self, df: PandasFrame):
        t = df[-1:]['time']
        self._last_timestamp = t.dt.to_pydatetime()[0]
        print(f'last timestamp = {self._last_timestamp}')

    def _get_ping_history(self) -> PandasFrame:
        df = super()._get_ping_history()
        self._update_last_timestamp(df)
        return df

    def _get_ping_newdata(self) -> PandasFrame:
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
        """ update chart based on last data (inserted after last update)
        """
        df = self._get_ping_newdata()
        df = self._prepare_data_heatmap(df)
        print('update:  df.shape=', df.shape)
        fig = self._create_heatmap(df)
        r = py.iplot(fig, filename=chart_name, fileopt='extend')
        print(f'       =========== {df.shape[0]} timemarks added to {r.resource}')


if __name__ == '__main__':
    p = _PltPingNotUpdatable()
    p.draw_heat()
    # p.draw_lines()
    #
    # iter_numb = 0
    # for iter_ in range(iter_numb):
    #     print(f'iter={iter_}/{iter_numb}')
    #     time.sleep(100)
    #     plt.update()
