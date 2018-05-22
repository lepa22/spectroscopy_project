import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import display


class PrinCompAn(object):
    "PCA class"

    def __init__(self, df, n_components=None):
        """
        """
        self.df = df
        self.n_components = n_components

        self.pca = PCA(self.n_components)
        self.pca_transformed = self.pca.fit_transform(self.df)
        self.n_comps_ = self.pca.n_components_
        self.explain()
        self.loadings()
        self.scores()

    def scores(self):
        """
        """
        scores = pd.DataFrame(self.pca_transformed, index=self.df.index,
                              columns=['PC{}'.format(i + 1) for i in
                                       range(self.pca_transformed.shape[1])])
        return scores

    def loadings(self):
        """
        """
        loadings = pd.DataFrame(self.pca.components_.T *
                                np.sqrt(self.exp_var_),
                                columns=[f'PC{i + 1}' for i in
                                         range(len(self.exp_var_))])
        return loadings

    def explain(self):
        """
        """
        self.exp_var_ = self.pca.explained_variance_
        self.cum_exp_var_ = np.cumsum(self.exp_var_)
        self.exp_var_rat_ = self.pca.explained_variance_ratio_
        self.cum_exp_var_rat_ = np.cumsum(self.exp_var_rat_)

        data = {'Explained variance': self.exp_var_,
                'Cumulative explained variance': self.cum_exp_var_,
                'Explained variance ratio': self.exp_var_rat_,
                'Cumulative explained variance ratio': self.cum_exp_var_rat_}

        return pd.DataFrame(data,
                            index=['PC{}'.format(i + 1) for i in
                                   range(self.pca_transformed.shape[1])],
                            columns=data.keys()).T

    def scree_plot(self, figsize=None, show='all', color=None,
                   title='Scree plot',
                   xlabel='Principal components', ylabel='Explained variance',
                   legend=True, legend_pos=(1, 0.85)):
        """
        """
        exp_var_rat_ = self.exp_var_rat_
        cum_exp_var_rat_ = self.cum_exp_var_rat_

        x = [i for i in range(len(exp_var_rat_))]

        fig, ax = plt.subplots(figsize=figsize)

        ax.set_ylim(-2.5, 110)

        yticks = ax.get_yticks()
        offset = (yticks[-1] - yticks[-2]) * 0.1

        if show == 'all':
            if color is None:
                color = [None] * 2

            ax.plot(x, exp_var_rat_ * 100, color=color[0], marker='o',
                    label='Explained variance')

            ax.plot(x, cum_exp_var_rat_ * 100, color=color[1], marker='s',
                    label='Cumulative explained variance')

            for i in range(0, len(exp_var_rat_)):
                ax.text(x[i], exp_var_rat_[i] * 100 + offset,
                        '{:.2f}%'.format(exp_var_rat_[i] * 100),
                        ha='center', va='bottom', fontsize=10)

            for i in range(1, len(cum_exp_var_rat_)):
                ax.text(x[i], cum_exp_var_rat_[i] * 100 + offset,
                        '{:.2f}%'.format(cum_exp_var_rat_[i] * 100),
                        ha='center', va='bottom', fontsize=10)

        elif show == 'exp_var_':
            ax.plot(x, exp_var_rat_ * 100, color=color, marker='o',
                    label='Explained variance')

            for i in range(0, len(exp_var_rat_)):
                ax.text(x[i], exp_var_rat_[i] * 100 + offset,
                        '{:.2f}%'.format(exp_var_rat_[i] * 100),
                        ha='center', va='bottom', fontsize=10)

        elif show == 'cum_exp_var_':
            ax.plot(x, cum_exp_var_rat_ * 100, color=color, marker='o',
                    label='Cumulative explained variance')

            for i in range(0, len(cum_exp_var_rat_)):
                ax.text(x[i], cum_exp_var_rat_[i] * 100 + offset,
                        '{:.2f}%'.format(cum_exp_var_rat_[i] * 100),
                        ha='center', va='bottom', fontsize=10)

        else:
            raise ValueError("show must be 'all',"
                             "'exp_var_' or 'cum_exp_var_'")

        plt.title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xticks(x)
        ax.set_xticklabels(['PC{:d}'.format(i + 1) for i in
                           range(len(exp_var_rat_))])

        if legend is True:
            plt.legend(facecolor='white', framealpha=0.75,
                       bbox_to_anchor=legend_pos)
        plt.tight_layout()

        plt.close()

        return fig

    def scores_plot(self, figsize=None, pc=[1, 2], size=100, color=None,
                    title='Scores plot', marker='o', text_offset=None,
                    text=True):
        """
        """
        fig, ax = plt.subplots(figsize=figsize)
        x = self.scores()
        names = x.index

        if type(color) is list:
            for i, name in enumerate(names):
                ax.scatter(x['PC{}'.format(pc[0])][x.index == name],
                           x['PC{}'.format(pc[1])][x.index == name],
                           marker=marker, s=size, color=color[i], alpha=0.75)
        else:
            for name in names:
                ax.scatter(x['PC{}'.format(pc[0])][x.index == name],
                           x['PC{}'.format(pc[1])][x.index == name],
                           marker=marker, s=size, color=color, alpha=0.75)

        ax.margins(0.1)

        if text is True:
            if text_offset is None:
                yticks = ax.get_yticks()
                text_offset = (yticks[-1] - yticks[-2]) * 0.15

            for name in names:
                ax.text(x['PC{}'.format(pc[0])][x.index == name],
                        x['PC{}'.format(pc[1])][x.index == name] + text_offset,
                        s=name, fontsize=10, alpha=0.75,
                        ha='center', va='bottom')

        plt.title(title)
        ax.set_xlabel('PC{} ({:.2f}%)'.format(pc[0],
                      self.exp_var_rat_[0] * 100))
        ax.set_ylabel('PC{} ({:.2f}%)'.format(pc[1],
                      self.exp_var_rat_[1] * 100))

        plt.tight_layout()

        plt.close()

        return fig

    def loadings_plot(self, pc=1, figsize=None, color=None,
                      title='Loadings plot',
                      xlabel=r'Raman shift (cm$^{-1}$)', ylabel='',
                      legend=False, **plot_kw):
        """
        """
        loadings = self.loadings().values

        fig = plt.figure(figsize=figsize)

        plt.plot(self.df.columns, loadings[:, pc - 1], color=color, **plot_kw)

        plt.title(title)
        plt.xlabel(xlabel)
        if ylabel == '':
            plt.ylabel(f'PC{pc}')
        else:
            plt.ylabel(ylabel)

        if legend is True:
            plt.legend(facecolor='white', framealpha=0.75)
        plt.tight_layout()

        plt.close()

        return fig

    def loadings_plot_all(self):
        for i in range(self.n_comps_):
            display(self.loadings_plot(figsize=(13, 6), pc=i + 1))
