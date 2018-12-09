import datetime
import time

import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py

from my_db import MyDb
import utils

chart_name = "ping"

host_info = {
    "www.ua": ["External", 4],
    "192.168.1.1": ["Router", 3],
    "192.168.1.64": ["old_hik", 2],
    "192.168.1.70": ["bullet", 1],
    "192.168.1.165": ["door_bell", 0],
}


# noinspection PyUnresolvedReferences
class PltPingNotUpdatable:
    """
    plot on plot.ly incrementally: init/update chart
    """
    df_hosts = None

    def __init__(self):
        self.db = MyDb()  # connection to MySQL

    def _get_ping_history(self) -> pd.core.frame.DataFrame:
        query = "select * from ping order by time asc"
        df = pd.read_sql(query, self.db.mydb)
        return df

    @classmethod
    def _prepare_data(cls, df: pd.core.frame.DataFrame):
        df.index = df['time']
        # del df['time']
        df.index = df.index.floor('1T')  # truncate to minutes

        # hosts --> columns
        df = df.groupby([df.index, 'host'])['avg_rtt'].agg('mean').unstack()  # minutes * hosts [avg_rtt]

        # cut peaks
        df[df > 8.] = 8.

        # sort columns by seqn from host_info
        df.rename(lambda s: f"{host_info[s][1]:02d}.{s}", axis="columns", inplace=True)  # host -> <seqn>.host
        df.sort_index(axis=1, inplace=True)  # sort by seqn
        df.rename(lambda s: host_info[s[3:]][0], axis="columns", inplace=True)  # <seqn>.host -> host_name

        return df

    @classmethod
    def _create_fig(cls, df):
        data = [
            go.Heatmap(
                z=df.T.values.tolist(),
                x=df.index.tolist(),
                y=df.columns,
                colorscale='Viridis',
            )
        ]
        layout = go.Layout(
            title='ping stat',
            xaxis=dict(ticks=''),  # nticks=12),
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
        print(f"plot created at {r.resource}.")
        globals()['df'] = df  # to check and play in console
        globals()['d'] = df_p


# noinspection PyUnresolvedReferences
class PltPing(PltPingNotUpdatable):
    def __init__(self):
        super().__init__()
        self._last_timestamp: datetime.datetime = None  # last timestamp been sent to chart server

    def _update_last_timestamp(self, df: pd.core.frame.DataFrame):
        t = df[-1:]['time']
        self._last_timestamp = t.dt.to_pydatetime()[0]
        print(f"last timestamp = {self._last_timestamp}")

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
        print(f"       =========== {df.shape[0]} timemarks added to {r.resource}")


if __name__ == "__main__":
    # plt = PltPing()
    plt = PltPing()
    plt.start()

    iter_numb = 0
    for iter_ in range(iter_numb):
        print(f"iter={iter_}/{iter_numb}")
        time.sleep(100)
        plt.update()
