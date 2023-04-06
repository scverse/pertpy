import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text
from matplotlib import pyplot
from scipy.stats import stats
from scvi import REGISTRY_KEYS


class JaxscgenPlot:
    """Plotting functions for Jaxscgen."""

    @staticmethod
    def reg_mean_plot(
        adata,
        condition_key,
        axis_keys,
        labels,
        path_to_save="./reg_mean.pdf",
        save=True,
        gene_list=None,
        show=False,
        top_100_genes=None,
        verbose=False,
        legend=True,
        title=None,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        **kwargs,
    ):
        """Plots mean matching figure for a set of specific genes.

        Args:
            adata:  AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
                    AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
                    corresponding to batch and cell type metadata, respectively.
            condition_key: The key for the condition
            axis_keys: Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
                       `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
            labels: Dictionary of axes labels of the form `{"x": "x-axis-name", "y": "y-axis name"}`.
            path_to_save: path to save the plot.
            save: Specify if the plot should be saved or not.
            gene_list: list of gene names to be plotted.
            show: if `True`: will show to the plot after saving it.
            **kwargs:
        """
        import seaborn as sns

        sns.set()
        sns.set(color_codes=True)

        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = np.asarray(np.mean(ctrl_diff.X, axis=0)).ravel()
            y_diff = np.asarray(np.mean(stim_diff.X, axis=0)).ravel()
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(x_diff, y_diff)
            if verbose:
                print("top_100 DEGs mean: ", r_value_diff**2)
        x = np.asarray(np.mean(ctrl.X, axis=0)).ravel()
        y = np.asarray(np.mean(stim.X, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print("All genes mean: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="black"))
                pyplot.plot(x_bar, y_bar, "o", color="red", markersize=5)
                # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # pyplot.text(x_bar, y1_bar, i, fontsize=11, color="black")
        if gene_list is not None:
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
                force_points=(0.0, 0.0),
            )
        if legend:
            pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title("", fontsize=fontsize)
        else:
            pyplot.title(title, fontsize=fontsize)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= " + f"{r_value_diff ** 2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )
        if save:
            pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        if show:
            pyplot.show()
        pyplot.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2

    @staticmethod
    def reg_var_plot(
        adata,
        condition_key,
        axis_keys,
        labels,
        path_to_save="./reg_var.pdf",
        save=True,
        gene_list=None,
        top_100_genes=None,
        show=False,
        legend=True,
        title=None,
        verbose=False,
        x_coeff=0.30,
        y_coeff=0.8,
        fontsize=14,
        **kwargs,
    ):
        """Plots variance matching figure for a set of specific genes.

        Args:
            adata: AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
                   AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
                   corresponding to batch and cell type metadata, respectively.
            condition_key: Key of the condition.
            axis_keys: Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
                       `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
            labels: Dictionary of axes labels of the form `{"x": "x-axis-name", "y": "y-axis name"}`.
            path_to_save: path to save the plot.
            save: Specify if the plot should be saved or not.
            gene_list: list of gene names to be plotted.
            show: if `True`: will show to the plot after saving it.
        """
        import seaborn as sns

        sns.set()
        sns.set(color_codes=True)

        sc.tl.rank_genes_groups(adata, groupby=condition_key, n_genes=100, method="wilcoxon")
        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = np.asarray(np.var(ctrl_diff.X, axis=0)).ravel()
            y_diff = np.asarray(np.var(stim_diff.X, axis=0)).ravel()
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(x_diff, y_diff)
            if verbose:
                print("Top 100 DEGs var: ", r_value_diff**2)
        if "y1" in axis_keys.keys():
            real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        x = np.asarray(np.var(ctrl.X, axis=0)).ravel()
        y = np.asarray(np.var(stim.X, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print("All genes var: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        # _p1 = pyplot.scatter(x, y, marker=".", label=f"{axis_keys['x']}-{axis_keys['y']}")
        # pyplot.plot(x, m * x + b, "-", color="green")
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        if "y1" in axis_keys.keys():
            y1 = np.asarray(np.var(real_stim.X, axis=0)).ravel()
            _ = pyplot.scatter(
                x,
                y1,
                marker="*",
                c="grey",
                alpha=0.5,
                label=f"{axis_keys['x']}-{axis_keys['y1']}",
            )
        if gene_list is not None:
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                pyplot.text(x_bar, y_bar, i, fontsize=11, color="black")
                pyplot.plot(x_bar, y_bar, "o", color="red", markersize=5)
                if "y1" in axis_keys.keys():
                    y1_bar = y1[j]
                    pyplot.text(x_bar, y1_bar, "*", color="black", alpha=0.5)
        if legend:
            pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if title is None:
            pyplot.title("", fontsize=12)
        else:
            pyplot.title(title, fontsize=12)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= " + f"{r_value_diff ** 2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )

        if save:
            pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        if show:
            pyplot.show()
        pyplot.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2

    @staticmethod
    def binary_classifier(
        scgen,
        adata,
        delta,
        ctrl_key,
        stim_key,
        path_to_save,
        save=True,
        fontsize=14,
    ):
        """Latent space classifier.

        Builds a linear classifier based on the dot product between
        the difference vector and the latent representation of each
        cell and plots the dot product results between delta and latent representation.

        Args:
            scgen: ScGen object that was trained.
            adata: AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
                   AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
                   corresponding to batch and cell type metadata, respectively.
            delta: Difference between stimulated and control cells in latent space
            ctrl_key: Key for `control` part of the `data` found in `condition_key`.
            stim_key: Key for `stimulated` part of the `data` found in `condition_key`.
            path_to_save: Path to save the plot.
            save: Specify if the plot should be saved or not.
            fontsize: Set the font size of the plot.
        """
        # matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        pyplot.close("all")
        adata = scgen._validate_anndata(adata)
        condition_key = scgen.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).original_key
        cd = adata[adata.obs[condition_key] == ctrl_key, :]
        stim = adata[adata.obs[condition_key] == stim_key, :]
        all_latent_cd = scgen.get_latent_representation(cd.X)
        all_latent_stim = scgen.get_latent_representation(stim.X)
        dot_cd = np.zeros(len(all_latent_cd))
        dot_sal = np.zeros(len(all_latent_stim))
        for ind, vec in enumerate(all_latent_cd):
            dot_cd[ind] = np.dot(delta, vec)
        for ind, vec in enumerate(all_latent_stim):
            dot_sal[ind] = np.dot(delta, vec)
        pyplot.hist(
            dot_cd,
            label=ctrl_key,
            bins=50,
        )
        pyplot.hist(dot_sal, label=stim_key, bins=50)
        # pyplot.legend(loc=1, prop={'size': 7})
        pyplot.axvline(0, color="k", linestyle="dashed", linewidth=1)
        pyplot.title("  ", fontsize=fontsize)
        pyplot.xlabel("  ", fontsize=fontsize)
        pyplot.ylabel("  ", fontsize=fontsize)
        pyplot.xticks(fontsize=fontsize)
        pyplot.yticks(fontsize=fontsize)
        ax = pyplot.gca()
        ax.grid(False)

        if save:
            pyplot.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
        pyplot.show()
