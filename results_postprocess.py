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

OUTPATH = "plots"
XLS = "BBC vs MABT.xlsx"

if __name__ == "__main__":
    sim_df = pd.read_excel(io=XLS, sheet_name="sim_results")
    real_df = pd.read_excel(io= XLS, sheet_name="real_datasets_summaries")

    ####
    figsize = (8.75, 2.75)
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

    # # Scatter 1
    # fig, axes = plt.subplots(1, 2, figsize=figsize)
    # for ax, x, data in zip(axes, (df_test_sim, df_test), ("Simulation", "Benchmakrs")):
    #     sns.scatterplot(x, x='_t', y='_95', hue='method', style='Reject', s=120, alpha=0.9, ax=ax)
    #     ax.axhline(0.95, linestyle='-.', color='red')
    #     ax.set_xlabel("Tightness")
    #     ax.set_ylabel("Coverage Probability")
    #     ax.set_title(f"{data}")
    #     ax.set_ylim((0.7, 1.01))
    #     ax.set_xlim((-0.0))
    #     ax.grid(True)
    # fig.tight_layout()
    # fig.savefig(os.path.join(OUTPATH, 'scatter_t_95.pdf'), bbox_inches='tight')

    # Swarm
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, x, data in zip(axes, (df_test_sim, df_test), ("Simulation", "Benchmakrs")):
        sns.swarmplot(x, x='method', y='_t', hue='Reject', s=5, ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel("Tightness")
        ax.set_title(f"{data}")
        ax.grid(True)
        sns.move_legend(ax, "lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPATH, 'swarm_2.pdf'), bbox_inches='tight')

    # Swarm ratio
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, x, data in zip(axes, (df_test_sim, df_test), ("Simulation", "Benchmakrs")):
        sns.swarmplot(x.drop(x[x['method'] == 'BBC'].index), x='method', y='ratio', hue='Reject', s=4.5, ax=ax)
        ax.set_xlabel(None)
        ax.set_ylabel("Tightness ratio to BBC")
        ax.set_title(f"{data}")
        ax.grid(True)
        sns.move_legend(ax, "lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPATH, 'swarm_ratio.pdf'), bbox_inches='tight')

    # BAR
    idx_order = df_test['method'].unique()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    width = 0.4
    for ax, df, data in zip(axes, (df_test_sim, df_test), ("Simulation", "Benchmakrs")):
        nonreject_percetage = 100- df.groupby(['method'])['Reject'].mean() * 100
        t_mean = df.loc[~df['Reject']].groupby('method')['_t'].mean()
        t_median = df.loc[~df['Reject']].groupby('method')['_t'].median()
        t_std = df.loc[~df['Reject']].groupby('method')['_t'].std()
        t_var = df.loc[~df['Reject']].groupby('method')['_t'].var()

        for x in np.setdiff1d(nonreject_percetage.index.to_list(), t_mean.index.to_list()):
            t_mean[x], t_median[x],  = 0, 0
        t_median = t_median.reindex(nonreject_percetage.index)
        t_mean = t_mean.reindex(nonreject_percetage.index)

        # Median and percetage
        ax2 = ax.twinx()
        plt2 = t_mean.loc[idx_order].plot(kind='bar', color='royalblue', ax=ax2, width=width, position=0, hatch='xxx',
                      yerr=t_var, capsize=4, label='Tightness in not rejected cases', edgecolor='black', linewidth=0.5)
        plt1 = nonreject_percetage.loc[idx_order].plot(kind='bar', color='orangered', ax=ax, width=width, position=1, hatch='///',
                                        label='Percentage of not rejected cases', edgecolor='black', linewidth=0.5)
        ax.set_xlabel(None)
        ax.set_xlim((-0.5, len(np.unique(t_mean.index))-0.5))
        ax.set_ylabel('Percentage', color='red')
        ax2.set_ylabel('Tightness', color='blue')
        ax.set_title(f"{data}")
        ax.grid(True)
    # legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='lower center', bbox_to_anchor=(0.5, -0.06),
      fancybox=True, shadow=True, ncol=2)
        #ax.legend(h1 + h2, l1 + l2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPATH, 'barplot_mean.pdf'), bbox_inches='tight')

    # # Mean std
    # fig, ax1 = plt.subplots(1,1,figsize=(7, 4))
    # ax2 = ax1.twinx()
    # nonreject_percetage.plot(kind='bar', color='red', ax=ax1, width=width, position=1, hatch='//')
    # t_mean.plot(kind='bar', color='blue', ax=ax2, width=width, position=0, hatch='xx',
    #               yerr=df_test.loc[df_test['Reject']].groupby('method')['_t'].std(), capsize=5)
    # ax1.set_xlabel('Methods')
    # ax1.set_xlim((-0.5, len(np.unique(nonreject_percetage.index))-0.5))
    # ax1.set_ylabel('Coverage prob. percetange Reject', color='red')
    # ax2.set_ylabel('Tightness', color='blue')
    # fig.tight_layout()
    # plt.savefig(os.path.join(OUTPATH, 'barplot_mean.pdf'))

    print('done')

    # #%
    # #################################################################################################
    # # OLD PLOTS
    # ATTRIBUTE = ('lower', '_95', '_t', '_corrected')
    # ATTRIBUTE_NAME = ('Lower confidence bound', 'Coverage probability', 'Tightness', 'Tightness corrected')
    # for att, att_name in zip(ATTRIBUTE, ATTRIBUTE_NAME):
    #     sim = sim_df.iloc[:, sim_df.columns.str.endswith(att)]
    #     real = real_df.iloc[:, real_df.columns.str.endswith(att)]
    #
    #     sim.columns = [x.split("_")[0] for x in sim.columns]
    #     real.columns = [x.split("_")[0] for x in real.columns]
    #
    #     cat = pd.concat([sim, real])
    #
    #     #############################################################################################################
    #     # BOXPLOT
    #     #############################################################################################################
    #     print(f"BOXPLOTS - {att_name}")
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    #     sns.boxplot(sim.dropna(axis=0), ax=ax[0])
    #     #sns.stripplot(data=sim, ax=ax[0], alpha=0.65, color='grey')
    #     ax[0].yaxis.grid(True)
    #     ax[0].set_title(' '.join([att_name, 'Simulation']))
    #
    #     sns.boxplot(real.dropna(axis=0), ax=ax[1])
    #     #sns.stripplot(data=real, ax=ax[1], alpha=0.65, color='grey')
    #     ax[1].yaxis.grid(True)
    #     ax[1].set_title(' '.join([att_name, 'Real-world data']))
    #
    #     # # Both, sim and real combined
    #     # sns.boxplot(cat.dropna(axis=0), ax=ax[2])
    #     # #sns.stripplot(data=cat.dropna(axis=0), ax=ax[2], alpha=0.65, color='grey')
    #     # ax[2].yaxis.grid(True)
    #     # ax[2].set_title(att_name)
    #
    #     # Cases
    #     if att == "95":
    #         for x in ax:
    #             x.set_ylim((0.59, 1.01))
    #             x.axhline(0.95, linestyle='--', c='r')
    #
    #
    #     fig.tight_layout()
    #     fig.savefig(os.path.join(OUTPATH, att_name+'.pdf'))
    #     plt.close('all')
    #
    #     #############################################################################################################
    #     # # Only combined
    #     #############################################################################################################
    #     # plt.figure(figsize=(8, 4))
    #     # sns.boxplot(cat.dropna(axis=0))
    #     # if att == "95":
    #     #     plt.axhline(0.95, linestyle='--', c='r')
    #     #     plt.ylim((0.59, 1.01))
    #     # plt.grid(True)
    #     # plt.tight_layout()
    #     # plt.savefig(os.path.join(OUTPATH, att_name+'_combined.png'))
    #
    #     #############################################################################################################
    #     # BARPLOT
    #     #############################################################################################################
    #     print(f"BARPLOTS - {att_name}")
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    #     sns.barplot(sim.dropna(axis=0), ax=ax[0], errorbar=None)
    #     ax[0].bar_label(ax[0].containers[0], fmt='%.3f', padding=1)
    #     ax[0].yaxis.grid(True)
    #     ax[0].set_title(' '.join([att_name, 'Simulation']))
    #
    #     sns.barplot(real.dropna(axis=0), ax=ax[1], errorbar=None)
    #     ax[1].bar_label(ax[1].containers[0], fmt='%.3f', padding=1)
    #     ax[1].yaxis.grid(True)
    #     ax[1].set_title(' '.join([att_name, 'Real-world data']))
    #
    #     # # Both, sim and real combined
    #     # sns.boxplot(cat.dropna(axis=0), ax=ax[2])
    #     # #sns.stripplot(data=cat.dropna(axis=0), ax=ax[2], alpha=0.65, color='grey')
    #     # ax[2].yaxis.grid(True)
    #     # ax[2].set_title(att_name)
    #
    #     # Cases
    #     if att == "95":
    #         for x in ax:
    #             x.set_ylim((0.59, 1.01))
    #             x.axhline(0.95, linestyle='--', c='r')
    #
    #     fig.tight_layout()
    #     fig.savefig(os.path.join(OUTPATH, att_name+'_BAR.pdf'))
    #     plt.close('all')
    #
    #


