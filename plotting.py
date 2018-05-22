import os
import matplotlib.pyplot as plt
import numpy as np


def isnotebook():
    """Check if script is running in interactive Jupyter notebook, qtconsole
    or standard Python interpreter to avoid drawing double plots in Jupyter
    notebook and qtconsole.
    Unfortunately, Jupyter console is interpreted as notebook, so plots won't
    display.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook, qtconsole or console
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def _simple_plot(x, y, figsize=None, title='', xlabel='', ylabel='',
                 xlim=None, ylim=None, legend=False, **plot_kw):
    """
    """
    fig = plt.figure(figsize=figsize)

    plt.plot(x, y, **plot_kw)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim(xlim)
    plt.ylim(ylim)

    if legend is True:
        plt.legend(facecolor='white')

    plt.tight_layout()

    if isnotebook():
        plt.close()
    else:
        plt.show()

    return fig


def _list_plot(x, y, figsize=None, title='', xlabel='', ylabel='',
               label='', linestyle='-', color=None, marker=None,
               xlim=None, ylim=None, legend=False, **plot_kw):
    """
    """
    fig = plt.figure(figsize=figsize)

    # check some line properties
    if type(linestyle) is str:
        linestyle = [linestyle] * len(y)

    if type(label) is str:
        label = [label] * len(y)

    if color is None:
        color = [None] * len(y)
    elif type(color) is str:
        color = [color] * len(y)

    if marker is None:
        marker = [None] * len(y)
    elif type(marker) is str:
        marker = [marker] * len(y)

    for i, j in enumerate(y):
        plt.plot(x, j, label=label[i], linestyle=linestyle[i],
                 color=color[i], marker=marker[i], **plot_kw)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim(xlim)
    plt.ylim(ylim)

    if legend is True:
        plt.legend(facecolor='white')

    plt.tight_layout()

    if isnotebook():
        plt.close()
    else:
        plt.show()

    return fig


def line_plot(x, y, figsize=None, title='', xlabel='', ylabel='', label='',
              linestyle='-', color=None, marker=None, xlim=None, ylim=None,
              legend=False, **plot_kw):
    """
    """
    if type(y) is list:
        return _list_plot(x, y, figsize=figsize, title=title, xlabel=xlabel,
                          ylabel=ylabel, label=label, linestyle=linestyle,
                          color=color, marker=marker, xlim=xlim, ylim=ylim,
                          legend=legend, **plot_kw)
    else:
        return _simple_plot(x, y, figsize=figsize, title=title, xlabel=xlabel,
                            ylabel=ylabel, xlim=xlim, ylim=ylim,
                            legend=legend, **plot_kw)


def scree_plot(y, figsize=None, show='all', color=None,
               title='Scree plot',
               xlabel='Principal components', ylabel='Explained variance',
               legend_pos=(1, 0.85)):
    """
    """
    exp_var_rat = y.explained_variance_ratio_
    cum_exp_var_rat = np.cumsum(exp_var_rat)

    x = [i for i in range(len(exp_var_rat))]

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_ylim(-2.5, 110)

    yticks = ax.get_yticks()
    offset = (yticks[-1] - yticks[-2]) * 0.1

    if show == 'all':
        if color is None:
            color = [None] * 2

        ax.plot(x, exp_var_rat * 100, color=color[0], marker='o',
                label='Explained variance')

        ax.plot(x, cum_exp_var_rat * 100, color=color[1], marker='s',
                label='Cumulative explained variance')

        for i in range(0, len(exp_var_rat)):
            ax.text(x[i], exp_var_rat[i] * 100 + offset,
                    '{:.2f}%'.format(exp_var_rat[i] * 100),
                    ha='center', va='bottom', fontsize=10)

        for i in range(1, len(cum_exp_var_rat)):
            ax.text(x[i], cum_exp_var_rat[i] * 100 + offset,
                    '{:.2f}%'.format(cum_exp_var_rat[i] * 100),
                    ha='center', va='bottom', fontsize=10)

    elif show == 'exp_var':
        ax.plot(x, exp_var_rat * 100, color=color, marker='o',
                label='Explained variance')

        for i in range(0, len(exp_va_rat)):
            ax.text(x[i], exp_var_rat[i] * 100 + offset,
                    '{:.2f}%'.format(exp_var[i] * 100),
                    ha='center', va='bottom', fontsize=10)

    elif show == 'cum_exp_var':
        ax.plot(x, cum_exp_var_rat * 100, color=color, marker='o',
                label='Explained variance')

        for i in range(0, len(cum_exp_var_rat)):
            ax.text(x[i], cum_exp_var_rat[i] * 100 + offset,
                    '{:.2f}%'.format(cum_exp_var_rat[i] * 100),
                    ha='center', va='bottom', fontsize=10)

    else:
        raise ValueError("show must be 'all', 'exp_var' or 'cum_exp_var'")

    plt.title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(x)
    ax.set_xticklabels(['PC{:d}'.format(i + 1) for i in range(len(exp_var))])

    plt.legend(facecolor='white', framealpha=0.75, bbox_to_anchor=legend_pos)
    plt.tight_layout()

    plt.close()

    return fig


def scores_plot(x, figsize=None, pc=[1, 2], size=100, color=None,
                title='Scores plot', marker='o', offset=None, text=True):
    """
    """
    fig, ax = plt.subplots(figsize=figsize)

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
        if offset is None:
            yticks = ax.get_yticks()
            offset = (yticks[-1] - yticks[-2]) * 0.15

        for name in names:
            ax.text(x['PC{}'.format(pc[0])][x.index == name],
                    x['PC{}'.format(pc[1])][x.index == name] + offset,
                    s=name, fontsize=10, alpha=0.75,
                    ha='center', va='bottom')

    plt.title(title)
    ax.set_xlabel('PC{} '
                  '({:.2f}%)'.format(pc[0],
                                     pca.explained_variance_ratio_[0] * 100))
    ax.set_ylabel('PC{} '
                  '({:.2f}%)'.format(pc[1],
                                     pca.explained_variance_ratio_[1] * 100))

    plt.tight_layout()

    plt.close()

    return fig


def saveplot(figure, path):
    """Save plot to the specified path.
    """
    # get directory and filename from path
    directory, filename = os.path.split(path)
    if directory == '':
        directory = '.'

    # check if directory exists. if not ask user whether to create it or not
    if not os.path.exists(directory):
        print('{} does not exist. '
              'Do you want to create it?'.format(directory))
        mkdir = ''
        while mkdir != 'y' and mkdir != 'n':
            mkdir = input('[y]es/[n]o: ')
            if mkdir == 'y':
                os.makedirs(directory)
            elif mkdir == 'n':
                print('Plot not saved.')
                return

    # path to save to
    savepath = os.path.join(directory, filename)

    # save figure
    figure.savefig(savepath)

    print('Plot saved as: {}'.format(savepath))
