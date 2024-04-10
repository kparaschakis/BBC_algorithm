import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend("Qt5Agg")
plt.rc('axes', axisbelow=True)
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font',**{'family':'serif','serif':['Times'], 'size': 10})
plt.rc('text', usetex=True)

HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

OUTPATH = "..\output"
XLS = "../output/final_results.xlsx"

if __name__ == "__main__":
    sim_df = pd.read_excel(io=XLS, sheet_name="sim_results")
    real_df = pd.read_excel(io=XLS, sheet_name="real_datasets_summaries")

    ####
    figsize = (8, 3)
    ATTRIBUTE = ('_95', '_pval', '_t')
    dfs = []
    dfs_sim = []
    for x in ATTRIBUTE:
        df_nonan = real_df.dropna(axis=0)
        df_sim_nonan = sim_df.dropna(axis=0)
        df_tmp = pd.melt(df_nonan, id_vars=['dataset'], value_vars=df_nonan.columns[df_nonan.columns.str.endswith(x)],
                         value_name=x, var_name='method')
        df_tmp_sim = pd.melt(df_sim_nonan, id_vars=['(alpha, beta)`', 'N', 'M', 'b', 'C'],
                             value_vars=df_sim_nonan.columns[df_sim_nonan.columns.str.endswith(x)], value_name=x, var_name='method')
        df_tmp['method'] = df_tmp['method'].str.split(x).str[0]
        df_tmp_sim['method'] = df_tmp_sim['method'].str.split(x).str[0]

        dfs.append(df_tmp)
        dfs_sim.append(df_tmp_sim)
    df_test = pd.merge(pd.merge(dfs[0], dfs[1]), dfs[2])
    df_test['Reject'] = df_test['_pval'] <= 0.05

    df_test_sim = pd.merge(pd.merge(dfs_sim[0], dfs_sim[1]), dfs_sim[2])
    df_test_sim['Reject'] = df_test_sim['_pval'] <= 0.05

    df_test['ratio'] = df_test['_t'] / df_test.groupby(['dataset'])['_t'].transform('first')
    df_test_sim['ratio'] = df_test_sim['_t'] / df_test_sim.groupby(['(alpha, beta)`', 'N', 'M', 'b', 'C'])['_t'].transform('first')

    df_test = df_test.replace("BBCF", "BBC-F")
    df_test_sim = df_test_sim.replace("BBCF", "BBC-F")

    # # Swarm
    # fig, axes = plt.subplots(1, 2, figsize=figsize)
    # for ax, x, data in zip(axes, (df_test_sim, df_test), ("Simulation", "Benchmarks")):
    #     sns.swarmplot(x, x='method', y='_t', hue='Reject', s=3.75, ax=ax)
    #     ax.set_xlabel(None)
    #     ax.set_ylabel("Tightness")
    #     ax.set_title(f"{data}")
    #     ax.grid(True)
    #     sns.move_legend(ax, "lower left")
    # fig.tight_layout()
    # fig.savefig(os.path.join(OUTPATH, 'swarm_2.pdf'), bbox_inches='tight')

    # Swarm ratio
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for ax, x, data in zip(axes, (df_test_sim, df_test), ("Simulation", "Benchmarks")):
        sns.swarmplot(x, x='method', y='ratio', hue='Reject', s=3, ax=ax)
        # Add grid at 1
        ax.axhline(1, color='gray', linewidth=0.5, alpha=0.3)
        yt = ax.get_yticks()
        yt = np.append(yt, 1)
        ax.set_yticks(yt)

        ax.set_xlabel(None)
        ax.set_ylabel("Tightness ratio to BBC")
        ax.set_title(f"{data}")
        ax.grid(True, which='major')
        sns.move_legend(ax, "upper right", bbox_to_anchor=(1.14, 1))
    #axes[1].legend().remove()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPATH, 'swarm_ratio_FULL.pdf'), bbox_inches='tight')

    # # BAR
    # idx_order = df_test['method'].unique()
    # fig, axes = plt.subplots(1, 2, figsize=figsize)
    # width = 0.4
    # for ax, df, data in zip(axes, (df_test_sim, df_test), ("Simulation", "Benchmakrs")):
    #     nonreject_percetage = 100- df.groupby(['method'])['Reject'].mean() * 100
    #     t_mean = df.loc[~df['Reject']].groupby('method')['_t'].mean()
    #     t_median = df.loc[~df['Reject']].groupby('method')['_t'].median()
    #     t_std = df.loc[~df['Reject']].groupby('method')['_t'].std()
    #     t_var = df.loc[~df['Reject']].groupby('method')['_t'].var()
    #
    #     for x in np.setdiff1d(nonreject_percetage.index.to_list(), t_mean.index.to_list()):
    #         t_mean[x], t_median[x],  = 0, 0
    #     t_median = t_median.reindex(nonreject_percetage.index)
    #     t_mean = t_mean.reindex(nonreject_percetage.index)
    #
    #     # Median and percetage
    #     ax2 = ax.twinx()
    #     plt2 = t_mean.loc[idx_order].plot(kind='bar', color='royalblue', ax=ax2, width=width, position=0, hatch='xxx',
    #                   yerr=t_var, capsize=4, label='Tightness in not rejected cases', edgecolor='black', linewidth=0.5)
    #     plt1 = nonreject_percetage.loc[idx_order].plot(kind='bar', color='orangered', ax=ax, width=width, position=1, hatch='///',
    #                                     label='Percentage of not rejected cases', edgecolor='black', linewidth=0.5)
    #     ax.set_xlabel(None)
    #     ax.set_xlim((-0.5, len(np.unique(t_mean.index))-0.5))
    #     ax.set_ylabel('Percentage', color='red')
    #     ax2.set_ylabel('Tightness', color='blue')
    #     ax.set_title(f"{data}")
    #     ax.grid(True)
    # # legend
    # h1, l1 = ax.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # fig.legend(h1 + h2, l1 + l2, loc='lower center', bbox_to_anchor=(0.5, -0.06),
    #   fancybox=True, shadow=True, ncol=2)
    #     #ax.legend(h1 + h2, l1 + l2)
    # fig.tight_layout()
    # fig.savefig(os.path.join(OUTPATH, 'barplot_mean.pdf'), bbox_inches='tight')

    print('done')

