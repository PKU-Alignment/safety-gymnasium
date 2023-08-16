# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Plotter class for plotting data from experiments."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from distutils.util import strtobool
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame


algo_map = {
    'happo': 'HAPPO',
    'mappo': 'MAPPO',
    'mappolag': 'MAPPOLag',
    'macpo': 'MACPO',
}


class Plotter:
    def __init__(self) -> None:
        """Initialize an instance of :class:`Plotter`."""
        self.div_line_width: int = 50
        self.exp_idx: int = 0
        self.units: dict[str, Any] = {}

    def plot_data(
        self,
        sub_figures: np.ndarray,
        data: list[DataFrame],
        xaxis: str = 'Steps',
        value: str = 'Rewards',
        condition: str = 'Condition1',
        smooth: int = 1,
        **kwargs: Any,
    ) -> None:
        if smooth > 1:
            y = np.ones(smooth)
            for datum in data:
                x = np.asarray(datum[value])
                z = np.ones(len(x))
                smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                datum[value] = smoothed_x
                if 'Unsafe Algorithms Costs' in datum:
                    x = np.asarray(datum['Unsafe Algorithms Costs'])
                    z = np.ones(len(x))
                    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                    datum['Unsafe Algorithms Costs'] = smoothed_x
                else:
                    x = np.asarray(datum['Safe Algorithms Costs'])
                    z = np.ones(len(x))
                    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                    datum['Safe Algorithms Costs'] = smoothed_x
        sns.set_palette('colorblind')

        current_palette = sns.color_palette()

        if isinstance(data, list):
            data_to_plot = pd.concat(data, ignore_index=True)
        sns.lineplot(
            data=data_to_plot,
            x=xaxis,
            y='Rewards',
            hue='Condition0',
            errorbar='sd',
            ax=sub_figures[0],
            palette=current_palette[0:4],
            **kwargs,
        )
        sns.lineplot(
            data=data_to_plot,
            x=xaxis,
            y='Unsafe Algorithms Costs',
            hue='Condition1',
            errorbar='sd',
            ax=sub_figures[1],
            palette=current_palette[0:2],
            **kwargs,
        )
        sns.lineplot(
            data=data_to_plot,
            x=xaxis,
            y='Safe Algorithms Costs',
            hue='Condition2',
            errorbar='sd',
            ax=sub_figures[2],
            palette=current_palette[2:4],
            **kwargs,
        )
        sub_figures[0].legend(
            loc='upper center',
            ncol=6,
            handlelength=1,
            mode='expand',
            borderaxespad=0.0,
            prop={'size': 13},
        )
        sub_figures[1].legend(
            loc='upper center',
            ncol=6,
            handlelength=1,
            mode='expand',
            borderaxespad=0.0,
            prop={'size': 13},
        )
        sub_figures[2].legend(
            loc='upper center',
            ncol=6,
            handlelength=1,
            mode='expand',
            borderaxespad=0.0,
            prop={'size': 13},
        )
        sub_figures[0].set_ylim(
            min(0, 1.1 * np.min(data_to_plot['Rewards'])), 1.1 * np.max(data_to_plot['Rewards'])
        )
        xscale = np.max(np.asarray(data_to_plot[xaxis], dtype=np.int32)) > 5e3
        if xscale:
            # just some formatting niceness: x-axis scale in scientific notation if max x is large
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        plt.tight_layout(pad=0.5)

    def get_datasets(
        self, logdir: str, condition: str | None = None, use_eval_result: bool = False
    ) -> list[DataFrame]:
        datasets: list[DataFrame] = []
        for root, _, files in os.walk(logdir):
            if 'progress.csv' in files:
                try:
                    with open(os.path.join(root, 'config.json'), encoding='utf-8') as f:
                        config = json.load(f)
                        if 'algorithm_name' in config:
                            algo_name = config['algorithm_name']
                except FileNotFoundError as error:
                    algo_name = root.split('/')[-2]
                algo_name = algo_map[algo_name]
                condition1 = condition or algo_name or 'exp'
                self.exp_idx += 1
                if condition1 not in self.units:
                    self.units[condition1] = 0
                unit = self.units[condition1]
                self.units[condition1] += 1
                try:
                    exp_data = pd.read_csv(os.path.join(root, 'progress.csv'))

                except FileNotFoundError as error:
                    progress_path = os.path.join(root, 'progress.csv')
                    raise FileNotFoundError(f'Could not read from {progress_path}') from error
                performance = (
                    'Metrics/EvalEpRet'
                    if 'Metrics/EvalEpRet' in exp_data and use_eval_result
                    else 'Metrics/EpRet'
                )
                cost_performance = (
                    'Metrics/EvalEpCost'
                    if 'Metrics/EvalEpCost' in exp_data and use_eval_result
                    else 'Metrics/EpCost'
                )
                exp_data.insert(len(exp_data.columns), 'Unit', unit)
                exp_data.insert(len(exp_data.columns), 'Condition0', condition1)

                if algo_name in ['MAPPO', 'HAPPO']:
                    if 'Condition1' not in exp_data.columns:
                        exp_data.insert(len(exp_data.columns), 'Condition1', algo_name)
                    exp_data.insert(
                        len(exp_data.columns),
                        'Unsafe Algorithms Costs',
                        exp_data[cost_performance].astype(np.float32),
                    )
                else:
                    if 'Condition2' not in exp_data.columns:
                        exp_data.insert(len(exp_data.columns), 'Condition2', algo_name)
                    exp_data.insert(
                        len(exp_data.columns),
                        'Safe Algorithms Costs',
                        exp_data[cost_performance].astype(np.float32),
                    )
                exp_data.insert(
                    len(exp_data.columns), 'Rewards', exp_data[performance].astype(np.float32)
                )

                total_steps = exp_data['Train/TotalSteps'].astype(np.float32)
                exp_data.insert(
                    len(exp_data.columns),
                    'Steps',
                    total_steps,
                )
                datasets.append(exp_data)
        return datasets

    def get_all_datasets(
        self,
        all_logdirs: list[str],
        legend: list[str] | None = None,
        select: str | None = None,
        exclude: str | None = None,
        use_eval_result: bool = False,
    ) -> list[DataFrame]:
        logdirs = []
        for logdir in all_logdirs:
            if osp.isdir(logdir) and logdir[-1] == os.sep:
                logdirs += [logdir]
            else:
                basedir = osp.dirname(logdir)
                prefix = logdir.split(os.sep)[-1]
                listdir = os.listdir(basedir)
                logdirs += sorted([osp.join(basedir, x) for x in listdir if prefix in x])
        if select is not None:
            logdirs = [log for log in logdirs if all(x in log for x in select)]
        if exclude is not None:
            logdirs = [log for log in logdirs if all(x not in log for x in exclude)]

        # verify logdirs
        print('Plotting from...\n' + '=' * self.div_line_width + '\n')
        for logdir in logdirs:
            print(logdir)
        print('\n' + '=' * self.div_line_width)

        # make sure the legend is compatible with the logdirs
        assert not (legend) or (
            len(legend) == len(logdirs)
        ), 'Must give a legend title for each set of experiments.'

        # load data from logdirs
        data = []
        if legend:
            for log, leg in zip(logdirs, legend):
                data += self.get_datasets(log, leg, use_eval_result=use_eval_result)
        else:
            for log in logdirs:
                data += self.get_datasets(log, use_eval_result=use_eval_result)
        return data

    # pylint: disable-next=too-many-arguments
    def make_plots(
        self,
        all_logdirs: list[str],
        legend: list[str] | None = None,
        xaxis: str = 'Steps',
        value: str = 'Rewards',
        count: bool = False,
        cost_limit: float = 25.0,
        smooth: int = 1,
        select: str | None = None,
        exclude: str | None = None,
        estimator: str = 'mean',
        save_dir: str = './',
        save_name: str | None = None,
        save_format: str = 'png',
        show_image: bool = False,
        use_eval_result: bool = False,
    ) -> None:
        assert xaxis is not None, 'Must specify xaxis'
        data = self.get_all_datasets(all_logdirs, legend, select, exclude, use_eval_result)
        condition = 'Condition2' if count else 'Condition1'
        # choose what to show on main curve: mean? max? min?
        estimator = getattr(np, estimator)
        sns.set()
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(21, 5),
        )
        self.plot_data(
            axes,
            data,
            xaxis=xaxis,
            value=value,
            condition=condition,
            smooth=smooth,
            estimator=estimator,
        )
        axes[1].axhline(y=cost_limit, ls='--', c='black', linewidth=2)
        axes[2].axhline(y=cost_limit, ls='--', c='black', linewidth=2)
        if save_name is None:
            save_name = all_logdirs[0].split('/')[-1]
        if show_image:
            plt.show()
        save_dir = save_dir.replace('runs', 'results')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(
            os.path.join(save_dir, f'figure_{save_name}.{save_format}'),
            bbox_inches='tight',
            pad_inches=0.0,
        )
        fig.savefig(
            os.path.join(save_dir, f'{save_name}.pdf'),
            bbox_inches='tight',
            pad_inches=0.0,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='Steps')
    parser.add_argument('--value', '-y', default='Rewards', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--estimator', default='mean')
    parser.add_argument('--use-eval-result', type=lambda x: bool(strtobool(x)), default=False)
    args = parser.parse_args()

    plotter = Plotter()
    listdir = os.listdir(args.logdir)
    for env in listdir:
        logdir = os.path.join(args.logdir, env)
        plotter.make_plots(
            [logdir],
            args.legend,
            args.xaxis,
            args.value,
            args.count,
            smooth=args.smooth,
            select=args.select,
            exclude=args.exclude,
            estimator=args.estimator,
            use_eval_result=args.use_eval_result,
            save_dir=args.logdir,
        )
