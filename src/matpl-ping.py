import datetime
import time
import sys
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

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

# @formatter:off
grade_info = [
    {'level': 'disconnect', 'left': -np.inf, 'right': -1,     'color_scale': [0.0, 'red']},
    {'level': 'poor',       'left': 7,       'right': np.inf, 'color_scale': [0.3, 'orange']},
    {'level': 'moderate',   'left': 5,       'right': 7,      'color_scale': [0.6, 'rgb(235,206,135)']},
    {'level': 'good',       'left': -1,      'right': 5,      'color_scale': [1.0, 'green']},
]
# @formatter:on

def _set_grade(df: PandasFrame):
    """ avg_rtt --> grade
    """
    df['grade'] = pd.Series(index=df.index)
    for gr in grade_info:
        df.loc[(gr['left'] < df.avg_rtt) & (df.avg_rtt <= gr['right']), 'grade'] = gr['color_scale'][0]
    print('grade: ', df['grade'].value_counts())


# noinspection PyUnresolvedReferences
class PltPing:
    """
    plot charts for ping data
    """
    def __init__(self):
        self.db = MyDb()  # connection to MySQL

    def _get_ping_history(self) -> PandasFrame:
        query = 'select * from ping order by time asc'
        df = pd.read_sql(query, self.db.mydb)
        return df

    @classmethod
    def _prepare_data(cls, df: PandasFrame):
        df.index = df['time']
        df.index = df.index.floor('1T')  # truncate to minutes

        df.avg_rtt.clip_upper(15.0, inplace=True)
        df.loc[df.avg_rtt == -1, 'avg_rtt'] = np.nan

        return df

    def draw_lines(self):
        df = self._get_ping_history()
        df = self._prepare_data(df)

        s = df.loc[df.host == 'www.ua', 'avg_rtt']
        globals()['s'] = s
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
        # hosts --> host_names
        df.columns = [host_info[d][0] for d in df.columns]
        globals()['dfm'] = df.copy()

        # @formatter:off
        scales = [
            {'period': 'hour',  'resample_step': None, 'ticks': 60, 'format':"%H:%M",    'nticks':30, 'xloc_base':3.0},
            {'period': 'day',   'resample_step': '1H', 'ticks': 24, 'format':"%Hh",      'nticks':30, 'xloc_base':1.0},
            {'period': 'month', 'resample_step': '1D', 'ticks': 30, 'format':"%Y-%m-%d", 'nticks':30, 'xloc_base':1.0},
        ]
        # @formatter:on

        for ind, sc in enumerate(scales):
            if sc['resample_step']:
                df2 = df.resample(sc['resample_step']).mean()
            else:
                df2 = df.copy()
            df2 = df2.tail(sc['ticks'])
            globals()[f"df_{sc['period']}"] = df2.copy()

            fig, ax = plt.subplots()
            ax.set_title(f"Last {sc['period']}")
            heatmap = ax.pcolor(df2.T,  # cmap=cmap,  #plt.cm.viridis,
                                cmap=LinearSegmentedColormap.from_list("", ["green", "olive", "darkkhaki", "orange"]),
                                vmin=np.nanmin(df2.T), vmax=np.nanmax(df2.T),
                                edgecolors='k', linewidth=1)  # cm.get_cmap('viridis', 256))
            ax.patch.set(color='red')  # hatch='x',edgecolor='red',fill=True,
            fig.colorbar(heatmap)  # , extend='both')

            # set x axis
            ax.xaxis.set_major_locator(ticker.IndexLocator(sc['xloc_base'], 0.5))
            xformatter = lambda x, pos: f"{df2.index[np.clip(int(x), 0, len(df2.index) - 1)].strftime(sc['format'])}"
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(xformatter))
            ax.tick_params(axis='x', labelrotation=45.)

            # set y axis
            ax.yaxis.set_major_locator(ticker.IndexLocator(1.0, 0.5))
            yformatter = lambda x, pos: f"{df2.columns[int(x)] if x < len(df2.columns) else '=No label='}"
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(yformatter))

            # show/save result
            plt.tight_layout()
            plt.savefig(fname=f"/home/im/mypy/pmon/tst/matpl-{sc['period']}.png")
            plt.show()
            globals()['a'] = ax


# noinspection PyUnresolvedReferences
class PltPingUpdatable(PltPing):
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
    p = PltPing()
    p.draw_heat()
    # p.draw_lines()
    #
    # iter_numb = 0
    # for iter_ in range(iter_numb):
    #     print(f'iter={iter_}/{iter_numb}')
    #     time.sleep(100)
    #     plt.update()
