from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd

from my_db import MyDb

chart_name = 'ping'

# @formatter:off
host_info = {
    'www.ua':        ['External',  0],
    '192.168.1.1':   ['Router',    1],
    '192.168.1.64':  ['old_hik',   4],
    '192.168.1.70':  ['bullet',    3],
    '192.168.1.165': ['door_bell', 2],
}
# @formatter:on

PandasFrame = pd.core.frame.DataFrame  # shortcut for annotations

# noinspection PyUnresolvedReferences
class PltPing:
    """
    plot charts for ping data
    """
    def __init__(self):
        self.db = None  # connection to MySQL
        # SQL ->(get_ping_history)-> df_inp ->(prepare_data)-> df ->(scale_df)-> df2 ->(draw..)-> charts
        self.df_inp: PandasFrame = None
        self.df: PandasFrame = None
        self.df2: PandasFrame = None
        globals()['p'] = self

    def _get_ping_history(self) -> PandasFrame:
        if not self.db:
            self.db = MyDb()
        query = 'select * from ping order by time asc'
        self.df_inp = pd.read_sql(query, self.db.mydb)
        return

    def load_data(self):
        if not self.df_inp:
            self._get_ping_history()
        self.df = self.df_inp

        # avg==-1 --> there was no answer from ping
        self.df.loc[self.df.avg_rtt == -1, 'avg_rtt'] = np.nan
        # df.avg_rtt.clip_upper(15.0, inplace=True)

        self.df.index = self.df['time']
        self.df.index = self.df.index.floor('1T')  # truncate to minutes

        # hosts --> columns
        self.df = self.df.groupby([self.df.index, 'host'])['ok', 'avg_rtt'].agg('mean').unstack()

        # sort columns (hosts_names) by host_info[host_name].seqn
        col = self.df.columns.tolist()
        # multiindex: level 0 - vars(ok,avg_rtt) level 1 - hosts;  reverse - to set external and router on top of chart
        col2 = sorted(col, key=lambda s: host_info[s[1]][1], reverse=True)
        self.df = self.df[col2]

        # host --> host_names
        self.df = self.df.rename(columns={h:host_info[h][0] for h in host_info})
        return

    # @formatter:off
    scales = {
        'hour':  {'resample_step': None, 'ticks': 60, 'format':"%H:%M",    'nticks':30, 'xloc_base':3.0},
        'day':   {'resample_step': '1H', 'ticks': 24, 'format':"%Hh",      'nticks':30, 'xloc_base':1.0},
        'month': {'resample_step': '1D', 'ticks': 30, 'format':"%Y-%m-%d", 'nticks':30, 'xloc_base':1.0},
    }
    # @formatter:on

    def scale_df(self, df: PandasFrame, period: str) -> PandasFrame:
        if self.scales[period]['resample_step']:
            self.df2 = df['ok'].resample(self.scales[period]['resample_step']).mean()
        else:
            self.df2 = df['ok'].copy()  # aggregate if need
        self.df2 = self.df2.tail(self.scales[period]['ticks'])  # cut data out of reporting period
        return self.df2

    def draw_heat_ok(self):
        """ Build heatmap.
        Indicator: bad pings % = count(ping<Thresh) / count(total pings)
        """
        self.load_data()
        for sc in self.scales:
            print("Chart for ", sc)
            self.df2 = self.scale_df(self.df, sc)

            fig, ax = plt.subplots()  # , sharex=True)
            ax.set_title(f"Last {sc}")

            color_list = ["green", "olive", "darkkhaki", "orange", "red"][::-1]  # reverse
            heatmap = ax.pcolor(self.df2.T,  # cm.get_cmap('viridis', 256))
                                cmap=LinearSegmentedColormap.from_list("", color_list),
                                vmin=np.nanmin(self.df2.T), vmax=np.nanmax(self.df2.T),
                                edgecolors='k', linewidth=1)
            # ax.patch.set(color='red')  # hatch='x',edgecolor='red',fill=True,

            b=fig.colorbar(heatmap)  # , extend='both')

            # set x axis
            ax.xaxis.set_major_locator(ticker.IndexLocator(self.scales[sc]['xloc_base'], 0.5))
            xformatter = lambda x, pos: f"{self.df2.index[np.clip(int(x), 0, len(self.df2.index)-1)].strftime(self.scales[sc]['format'])}"
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(xformatter))
            ax.tick_params(axis='x', labelrotation=90.)

            # set y axis
            ax.yaxis.set_major_locator(ticker.IndexLocator(1.0, 0.5))
            yformatter = lambda x, pos: f"{self.df2.columns[int(x)] if x < len(self.df2.columns) else '=No label='}"
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(yformatter))

            # show/save result
            # fig.subplots_adjust(hspace=0)
            # plt.tight_layout()
            fig.savefig(fname=f"/home/im/mypy/pmon/tst/heat-{sc}.png")
            fig.show()

    def draw_line_rtt(self):
        # ping line
        self.load_data()
        for ind, sc in enumerate(scales):
            # prepare data
            df3 = df['avg_rtt'].resample(sc['resample_step']).mean() if sc['resample_step'] else df[
                'avg_rtt'].copy()  # aggregate if need
            df3 = df3.tail(sc['ticks'])  # cut data out of reporting period
            df3[df3 == -1] = 100
            # hosts --> host_names
            df3.columns = [host_info[d][0] for d in df3.columns]
            globals()['d3'] = df3.copy()

            f2, ax_2 = plt.subplots()

            # draw
            # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
            ax_2.plot(x=df3.index, y=df3['External'])  # df3['External'].index,
            # ax2.xaxis.set_major_locator(ticker.IndexLocator(sc['xloc_base'], 0.5))
            # ax2.xaxis.set_major_locator(ticker.AutoLocator())
            # ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter("{}"))

            ax2.grid(True)


if __name__ == '__main__':
    p = PltPing()
    p.draw_heat_ok()
