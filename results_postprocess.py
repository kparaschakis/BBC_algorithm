import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend("Qt5Agg")
plt.rc('axes', axisbelow=True)
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('text', usetex=True)

OUTPATH = "plots"
XLS = "BBC vs MABT.xlsx"

if __name__ == "__main__":
    sim_df = pd.read_excel(io=XLS, sheet_name="sim_results")
    real_df = pd.read_excel(io= XLS, sheet_name="real_datasets_summaries")

    ATTRIBUTE = ('lower', '95', '_t', '_corrected')
    ATTRIBUTE_NAME = ('Lower confidence bound', 'Coverage probability', 'Tightness', 'Tightness corrected')

    # BIG LOOP
    for att, att_name in zip(ATTRIBUTE, ATTRIBUTE_NAME):
        sim = sim_df.iloc[:, sim_df.columns.str.endswith(att)]
        real = real_df.iloc[:, real_df.columns.str.endswith(att)]

        sim.columns = [x.split("_")[0] for x in sim.columns]
        real.columns = [x.split("_")[0] for x in real.columns]

        cat = pd.concat([sim, real])

        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        sns.boxplot(sim, ax=ax[0])
        #sns.stripplot(data=sim, ax=ax[0], alpha=0.65, color='grey')
        ax[0].yaxis.grid(True)
        ax[0].set_title(' '.join([att_name, 'Sim']))

        sns.boxplot(real, ax=ax[1])
        #sns.stripplot(data=real, ax=ax[1], alpha=0.65, color='grey')
        ax[1].yaxis.grid(True)
        ax[1].set_title(' '.join([att_name, 'Real']))

        sns.boxplot(cat.dropna(axis=0), ax=ax[2])
        #sns.stripplot(data=cat.dropna(axis=0), ax=ax[2], alpha=0.65, color='grey')
        ax[2].yaxis.grid(True)
        ax[2].set_title(att_name)

        # Cases
        if att == "95":
            for x in ax:
                x.set_ylim((0.59, 1.01))
                x.axhline(0.95, linestyle='--', c='r')

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPATH, att_name+'.png'))

        # Only combined
        plt.figure(figsize=(8, 4))
        sns.boxplot(cat.dropna(axis=0))
        if att == "95":
            plt.axhline(0.95, linestyle='--', c='r')
            plt.ylim((0.59, 1.01))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPATH, att_name+'_combined.png'))

    plt.show(block=True)

    #
    # # AUC-ROC Lowerbound
    # sim_data_low = sim_df.iloc[:, sim_df.columns.str.endswith('lower')].dropna()
    # sim_data_low.columns = [x.split("_")[0] for x in sim_data_low.columns]
    # real_data_low = real_df.iloc[:, real_df.columns.str.endswith('lower')].dropna()
    # real_data_low.columns = [x.split("_")[0] for x in real_data_low.columns]
    #
    #
    # sns.boxplot(real_data_low)
    #
    # plt.show(block=True)
    #
    #
    # # 95CI
    # sim_data_95 = sim_df.iloc[:, sim_df.columns.str.endswith('95')]
    # real_data_95 = real_df.iloc[:, real_df.columns.str.endswith('95')]
    # cat_95 = pd.concat([real_data_95, sim_data_95])
    #
    # plt.figure()
    # sns.boxplot(real_data_95.dropna())
    # plt.title('Real')
    #
    # plt.figure()
    # sns.boxplot(sim_data_95.dropna())
    # plt.title('Sim')
    #
    # plt.figure()
    # sns.boxplot(cat_95.dropna())
    #
    # plt.show(block=True)
    #
    # Tightness
    # sim_data_t = sim_df.iloc[:, sim_df.columns.str.endswith('_t')]
    # real_data_t = real_df.iloc[:, real_df.columns.str.endswith('_t')]
    # cat_t = pd.concat([sim_data_t, real_data_t])
    #
    # fig, ax = plt.subplots(1, 3, figsize=(15, 8))
    # sns.boxplot(sim_data_t, ax=ax[0])
    # ax[0].set_title("sim")
    #
    # sns.boxplot(real_data_t, ax=ax[1])
    # ax[1].set_title("real")
    #
    # sns.boxplot(cat_t, ax=ax[2])
    #
    # fig.tight_layout()
    # plt.show(block=True)


