import anndata as ad
import inspect
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.spatial import cKDTree
from pathlib import Path
import plotly.graph_objs as go
from matplotlib.ticker import FormatStrFormatter
import warnings

from spatial_compare.utils import grouped_obs_mean


DEFAULT_DATA_NAMES = ["Data 0", "Data 1"]


class SpatialCompare:
    """
    A class for comparing spatial data between two AnnData objects.

    Parameters:
    ----------
    ad_0 : AnnData
        The first AnnData object to compare.
    ad_1 : AnnData
        The second AnnData object to compare.
    data_names : list, optional
        A list of names for the two datasets, by default "Data 0" and "Data 1"
    obsm_key : str, optional
        The key for the spatial data in the `obsm` attribute of the AnnData objects, by default "spatial_cirro_grid".
    category : str, optional
        The category to compare in the `obs` attribute of the AnnData objects, by default "subclass".

    Attributes:
    ----------
    ad_0 : AnnData
        The first AnnData object.
    ad_1 : AnnData
        The second AnnData object.
    obsm_key : str
        The key for the spatial data in the `obsm` attribute.
    category : str
        The category to compare in the `obs` attribute.
    shared_genes : list
        The list of genes that are shared between the two datasets.
    can_compare : bool
        A flag indicating if the datasets can be compared.
    data_names : list
        A list of names for the two datasets.
    spatial_compare_results : None or dict
        The results of the spatial comparison.

    Methods:
    -------
    set_category(category)
        Set the category to compare.
    spatial_plot(plot_legend=True, min_cells_to_plot=10, decimate_for_spatial_plot=1, figsize=[20,10], category_values=[])
        Plot the spatial data for the two datasets.
    de_novo_cluster(plot_stuff=False, correspondence_level="leiden_1",rerun_preprocessing=False)
        Perform de novo clustering on the two datasets.
    find_matched_groups(n_top_groups=100, n_shared_groups=30, min_n_cells=100, category_values=[], exclude_group_string="zzzzzzzzzzzzzzz", plot_stuff=False, figsize=[10,10])
        Find matched groups between the two datasets.
    compare_expression(category_values=[], plot_stuff=False, min_mean_expression=.2, min_genes_to_compare=5, min_cells=10)
        Compare gene expression between the two datasets.

    """

    def __init__(
        self,
        ad_0: ad.AnnData,
        ad_1: ad.AnnData,
        data_names=DEFAULT_DATA_NAMES,
        obsm_key="spatial_cirro_grid",
        category="subclass",
    ):
        """
        Initialize the SpatialCompare object.

        Parameters:
        ----------
        ad_0 : AnnData
            The first AnnData object to compare.
        ad_1 : AnnData
            The second AnnData object to compare.
        data_names : list, optional
            A list of names for the two datasets, by default DEFAULT_DATA_NAMES.
        obsm_key : str, optional
            The key for the spatial data in the `obsm` attribute of the AnnData objects, by default "spatial_cirro_grid".
        category : str, optional
            The category to compare in the `obs` attribute of the AnnData objects, by default "subclass".
        """

    def __init__(
        self,
        ad_0: ad.AnnData,
        ad_1: ad.AnnData,
        data_names=DEFAULT_DATA_NAMES,
        obsm_key="spatial_cirro_grid",
        category="subclass",
    ):

        # validate self.ad_0 and self.ad_1:
        if not isinstance(ad_0, ad.AnnData) or not isinstance(ad_1, ad.AnnData):
            raise ValueError("ad_0 and self.ad_1 must be AnnData objects")

        self.ad_0 = ad_0
        self.ad_1 = ad_1

        # make sure both AnnData objects have raw counts in X:

        # validate spatial key in obsm
        if (
            obsm_key not in self.ad_0.obsm.keys()
            or obsm_key not in self.ad_1.obsm.keys()
        ):
            raise ValueError(
                " obsm key " + obsm_key + " is not in both input AnnData objects"
            )

        self.obsm_key = obsm_key

        # vaidate category in obs:

        if (
            category not in self.ad_0.obs.columns
            or category not in self.ad_1.obs.columns
        ):
            self.category = None
            self.can_compare = False
            warnings.warn(
                "category " + category + " is not in both input AnnData objects"
            )

        self.category = category

        for adobj in [self.ad_0, self.ad_1]:
            if "gene" not in adobj.var.columns:
                adobj.var["gene"] = adobj.var.index

        # identify the intersection of genes between the 2 datasets:
        self.shared_genes = list(set(self.ad_0.var.index) & set(self.ad_1.var.index))
        print(
            "input anndata objects have "
            + str(len(self.shared_genes))
            + " shared genes"
        )

        self.can_compare = True

        self.data_names = data_names

        self.spatial_compare_results = None

        self.ran_preprocessing = False

    def set_category(self, category):
        if (
            category not in self.ad_0.obs.columns
            or category not in self.ad_1.obs.columns
        ):
            self.category = None
            self.can_compare = False
            raise Warning(
                "category " + category + " is not in both input AnnData objects"
            )

        self.category = category
        self.can_compare = True

    def spatial_plot(
        self,
        plot_legend=True,
        min_cells_to_plot=10,
        decimate_for_spatial_plot=1,
        figsize=[20, 10],
        category_values=[],
    ):

        plt.figure(figsize=figsize)
        all_category_values = set(self.ad_0.obs[self.category].unique()) | set(
            self.ad_1.obs[self.category].unique()
        )
        if len(category_values) == 0:
            category_values = all_category_values

        for c in category_values:
            plt.subplot(1, 2, 1)
            plt.title(self.data_names[0])
            if np.sum(self.ad_0.obs[self.category] == c) > min_cells_to_plot:
                label = c + ": " + str(np.sum(self.ad_0.obs[self.category] == c))
            else:
                label = None

            plt.plot(
                self.ad_0.obsm[self.obsm_key][self.ad_0.obs[self.category] == c, 0][
                    ::decimate_for_spatial_plot
                ],
                self.ad_0.obsm[self.obsm_key][self.ad_0.obs[self.category] == c, 1][
                    ::decimate_for_spatial_plot
                ],
                ".",
                label=label,
                markersize=0.5,
            )
            plt.axis("equal")
            if plot_legend:
                plt.legend(markerscale=5)
            plt.subplot(1, 2, 2)
            plt.title(self.data_names[1])
            if np.sum(self.ad_1.obs[self.category] == c) > min_cells_to_plot:
                label = c + ": " + str(np.sum(self.ad_1.obs[self.category] == c))
            else:
                label = None
            plt.plot(
                self.ad_1.obsm[self.obsm_key][self.ad_1.obs[self.category] == c, 0][
                    ::decimate_for_spatial_plot
                ],
                self.ad_1.obsm[self.obsm_key][self.ad_1.obs[self.category] == c, 1][
                    ::decimate_for_spatial_plot
                ],
                ".",
                label=label,
                markersize=0.5,
            )
            plt.axis("equal")
            if plot_legend:
                plt.legend(markerscale=5)

    def de_novo_cluster(
        self, plot_stuff=False, correspondence_level="leiden_1", run_preprocessing=False
    ):

        # force running of preprocessing the first time through
        if not run_preprocessing:
            if not self.ran_preprocessing:
                run_preprocessing = True

        self.ad_0 = filter_and_cluster_twice(
            self.ad_0, plot_stuff=plot_stuff, run_preprocessing=run_preprocessing
        )
        self.ad_1 = filter_and_cluster_twice(
            self.ad_1, plot_stuff=plot_stuff, run_preprocessing=run_preprocessing
        )

        if run_preprocessing:
            self.ran_preprocessing = True
        find_best_match_groups(
            self.ad_0,
            self.ad_1,
            group_names=[correspondence_level, correspondence_level],
        )
        self.can_compare = True
        self.set_category("matched_" + correspondence_level)
        return True

    def find_matched_groups(
        self,
        n_top_groups=100,
        n_shared_groups=30,
        category_values=[],
        exclude_group_string="zzzzzzzzzzzzzzz",
        plot_stuff=False,
        figsize=[10, 10],
    ):
        """
        find groups of cells that:
        - are in the n_top_groups in population for both datasets
        - are up to n_shared_groups in number

        Parameters:
        ----------
        n_top_groups : int, optional
            The number of groups from each dataset, ranked by population.  default 100.
        n_shared_groups : int, optional
            The maximum number of shared groups to consider.  default 30.
        category_values : list, optional
            Override the previous parameters by supplying a list of category values.  default [].
        exclude_group_string : str, optional
            Can be used to filter the category values at the very beginning of the process,
              e.g. to exclude category values with "NN" in the name.  default "zzzzzzzzzzzzzzz".
        plot_stuff : bool, optional
            Whether to plot the results.  default False.
        """

        sorted_counts = {}
        _other_columns = [
            [c for c in self.ad_0.obs.columns if c != self.category][0],
            [c for c in self.ad_1.obs.columns if c != self.category][0],
        ]

        for ii, df in enumerate([self.ad_0.obs, self.ad_1.obs]):
            sorted_counts[ii] = (
                df.loc[:, [self.category, _other_columns[ii]]]
                .groupby(self.category, observed=True)
                .count()
                .sort_values(_other_columns[ii], ascending=False)
            )

        in_top_N_0 = (
            sorted_counts[0]
            .loc[
                [c for c in sorted_counts[0].index if exclude_group_string not in c], :
            ]
            .head(n_top_groups)
        )
        in_top_N_1 = (
            sorted_counts[1]
            .loc[
                [c for c in sorted_counts[1].index if exclude_group_string not in c], :
            ]
            .head(n_top_groups)
        )

        if len(category_values) > 0:
            in_top_N_0 = in_top_N_0.loc[in_top_N_0.index.isin(category_values), :]
            in_top_N_1 = in_top_N_1.loc[in_top_N_1.index.isin(category_values), :]

        shared_top = list(
            in_top_N_0.loc[in_top_N_0.index.isin(in_top_N_1.index), :].index
        )

        if len(shared_top) > n_shared_groups:
            shared_top = shared_top[: n_shared_groups + 1]

        prop_corr = np.corrcoef(
            in_top_N_0.loc[shared_top, :][_other_columns[0]],
            in_top_N_1.loc[shared_top, :][_other_columns[1]],
        )[1][0]

        if plot_stuff:
            plt.figure(figsize=figsize)
            plt.loglog(
                in_top_N_0.loc[shared_top, :][_other_columns[0]],
                in_top_N_1.loc[shared_top, :][_other_columns[1]],
                ".",
            )
            for g in shared_top:
                # long names are annoying to read
                if len(g) < 50:
                    plt.text(
                        in_top_N_0.loc[g, :][_other_columns[0]],
                        in_top_N_1.loc[g, :][_other_columns[1]],
                        g,
                    )
            plt.xlabel(self.data_names[0])
            plt.ylabel(self.data_names[1])
            plt.title("cell type abundances\n correlation = " + str(prop_corr)[:6])
            plt.plot(
                [
                    np.min(in_top_N_0.loc[shared_top, :][_other_columns[0]]),
                    np.max(in_top_N_0.loc[shared_top, :][_other_columns[0]]),
                ],
                [
                    np.min(in_top_N_0.loc[shared_top, :][_other_columns[0]]),
                    np.max(in_top_N_0.loc[shared_top, :][_other_columns[0]]),
                ],
                "--",
            )

            plt.axis("equal")
        return {
            "category_top_values": shared_top,
            "category": self.category,
            "proportion_correlation": prop_corr,
            "n0": in_top_N_0,
            "n1": in_top_N_1,
        }

    def compare_expression(
        self,
        category_values=[],
        plot_stuff=False,
        min_mean_expression=0.2,
        min_genes_to_compare=5,
        min_cells=10,
    ):
        # group cells
        if len(category_values) == 0:
            raise ValueError(
                "please supply a list of values for the category " + self.category
            )

        category_records = []
        gene_ratio_dfs = {}
        for category_value in category_values:
            group_mask_0 = self.ad_0.obs[self.category] == category_value
            group_mask_1 = self.ad_1.obs[self.category] == category_value

            if np.sum(group_mask_0) < min_cells or np.sum(group_mask_1) < min_cells:
                print(
                    "at least 1 input has less than "
                    + str(min_cells)
                    + " cells in "
                    + self.category
                    + " == "
                    + category_value
                )
                continue

            means_0 = np.array(
                np.mean(
                    self.ad_0[
                        group_mask_0, self.ad_0.var.index.isin(self.shared_genes)
                    ].X,
                    axis=0,
                )
            ).flatten()
            means_1 = np.array(
                np.mean(
                    self.ad_1[
                        group_mask_1, self.ad_1.var.index.isin(self.shared_genes)
                    ].X,
                    axis=0,
                )
            ).flatten()
            means_0_gt_min = np.nonzero(means_0 > min_mean_expression)[0]
            means_1_gt_min = np.nonzero(means_1 > min_mean_expression)[0]
            above_means0 = self.ad_0.var[
                self.ad_0.var.index.isin(self.shared_genes)
            ].iloc[means_0_gt_min]
            above_means1 = self.ad_1.var[
                self.ad_1.var.index.isin(self.shared_genes)
            ].iloc[means_1_gt_min]
            shared_above_mean = [
                g for g in above_means1.index if g in above_means0.index
            ]
            if len(shared_above_mean) < min_genes_to_compare:
                print(
                    self.category
                    + " "
                    + category_value
                    + " has less than "
                    + str(min_genes_to_compare)
                    + "\n shared genes above minimum mean = "
                    + str(min_mean_expression)
                )
                continue

            means_0 = np.array(
                np.mean(self.ad_0[group_mask_0, shared_above_mean].X, axis=0)
            ).flatten()
            means_1 = np.array(
                np.mean(self.ad_1[group_mask_1, shared_above_mean].X, axis=0)
            ).flatten()
            shared_genes = shared_above_mean
            p_coef = np.polynomial.Polynomial.fit(means_0, means_1, 1).convert().coef
            category_records.append(
                {
                    self.category: category_value,
                    "slope": p_coef[1],
                    "mean_ratio": np.mean(means_1 / means_0),
                    "correlation": np.corrcoef(means_0, means_1)[0][1],
                    "n_cells_0": np.sum(group_mask_0),
                    "n_cells_1": np.sum(group_mask_1),
                    "total_count_ratio": np.sum(self.ad_1[group_mask_1, shared_genes].X)
                    / np.sum(self.ad_0[group_mask_0, shared_genes].X),
                }
            )

            gene_ratio_dfs[category_value] = pd.DataFrame(
                means_1 / means_0,
                columns=[self.data_names[1] + " / " + self.data_names[0] + " ratio"],
                index=shared_genes,
            )

            if plot_stuff:
                plt.figure(figsize=[10, 10])
                plt.title(
                    self.category
                    + ": "
                    + category_value
                    + "\nmean counts per cell\ncorrelation: "
                    + str(category_records[-1]["correlation"])[:4]
                    + " mean ratio: "
                    + str(category_records[-1]["mean_ratio"])[:4]
                )
                low_expression = np.logical_and(means_0 < 1.0, means_1 < 1.0)
                plt.loglog(
                    means_0[low_expression],
                    means_1[low_expression],
                    ".",
                    color=[0.5, 0.5, 0.5],
                )
                plt.loglog(
                    means_0[np.logical_not(low_expression)],
                    means_1[np.logical_not(low_expression)],
                    ".",
                )

                plt.xlabel(self.data_names[0] + ", N = " + str(np.sum(group_mask_0)))
                plt.ylabel(self.data_names[1] + ", N = " + str(np.sum(group_mask_1)))

                for g in shared_genes:
                    if (
                        means_0[np.nonzero(np.array(shared_genes) == g)] == 0
                        or means_1[np.nonzero(np.array(shared_genes) == g)] == 0
                    ) or low_expression[np.array(shared_genes) == g]:
                        continue
                    plt.text(
                        means_0[np.nonzero(np.array(shared_genes) == g)],
                        means_1[np.nonzero(np.array(shared_genes) == g)],
                        g,
                        fontsize=10,
                    )
                plt.plot(
                    [np.min(means_0), np.max(means_0)],
                    [np.min(means_0), np.max(means_0)],
                    "--",
                )
        print(gene_ratio_dfs.keys())
        if len(gene_ratio_dfs.keys()) > 0:

            gene_ratio_df = pd.concat(gene_ratio_dfs, axis=1)
        else:
            gene_ratio_df = None
        return {
            "data_names": self.data_names,
            "category_results": pd.DataFrame.from_records(category_records),
            "gene_ratio_dataframe": gene_ratio_df,
        }

    def plot_detection_ratio(self, gene_ratio_dataframe, figsize=[15, 15]):

        detection_ratio_plots(
            gene_ratio_dataframe, data_names=self.data_names, figsize=figsize
        )

    def spatial_compare(self, **kwargs):

        if "category" in kwargs.keys():
            self.set_category(kwargs["category"])

        if not self.can_compare:
            raise ValueError(
                "cannot compare AnnData objects without a category.\n run de novo clustering or supply a category with set_category()"
            )

        fmr_kwargs = {
            key: kwargs[key]
            for key in inspect.signature(
                SpatialCompare.find_matched_groups
            ).parameters.keys()
            if key in kwargs.keys()
        }
        ce_kwargs = {
            key: kwargs[key]
            for key in inspect.signature(
                SpatialCompare.compare_expression
            ).parameters.keys()
            if key in kwargs.keys() and key != "category_values"
        }

        match_results = self.find_matched_groups(**fmr_kwargs)

        expression_results = self.compare_expression(
            category_values=match_results["category_top_values"], **ce_kwargs
        )
        match_results["expression_results"] = expression_results
        return match_results

    def run_and_plot(self, **kwargs):
        if "category" in kwargs.keys():
            self.set_category(kwargs["category"])

        self.spatial_plot()
        self.spatial_compare_results = self.spatial_compare(plot_stuff=True, **kwargs)
        self.plot_detection_ratio(
            self.spatial_compare_results["expression_results"]["gene_ratio_dataframe"],
            figsize=[30, 20],
        )
        return True

    def collect_mutual_match_and_doublets(
        self,
        bc,
        save=True,
        nn_dist=2.5,
        reuse_saved=True,
        savepath=None,
        min_transcripts=40,
    ):
        """
        Runs all relevant functions in required order, generating dataframe summarizing comparisons between two segmentations for a single section.
        INPUTS
        bc: unique identifier for section segmentation comparison was run on. String.
        seg_name_a, seg_name_b: names of segmentations to be compared, string
        save: whether to save the intermediate and final results. Bool, default true
        nn_dist: distance cutoff for identifying likely same cell between segmentations. float, default 2.5 (um)
        min_transcripts: minimum number of transcripts needed to define a cell too low quality to be considered for mapping
        savepath: path to which you'd like to save your results.
        OUTPUTS:
        seg_comp_df: dataframe with unique index describing cell spatial locations (x and y), segmentation identifier, low quality cell identifier, mutual matches, and putative doublets.
        """

        # grab base comparison df
        seg_comp_df = get_segmentation_data(
            bc,
            self.ad_0,
            self.ad_1,
            self.data_names[0],
            self.data_names[1],
            save,
            savepath,
            reuse_saved,
            min_transcripts,
        )

        # mutual matches with both segmentations filtered
        dfa = seg_comp_df[
            (seg_comp_df["source"] == self.data_names[0])
            & (seg_comp_df["low_quality_cells"] == False)
        ]
        dfb = seg_comp_df[
            (seg_comp_df["source"] == self.data_names[1])
            & (seg_comp_df["low_quality_cells"] == False)
        ]
        col_name_both_filt = (
            "match_" + self.data_names[0] + "_filt_" + self.data_names[1] + "_filt"
        )
        matched_dfa_only, matched_dfb_only = get_mutual_matches(dfa, dfb, nn_dist)
        seg_comp_df.loc[matched_dfb_only.index.values.tolist(), col_name_both_filt] = (
            matched_dfb_only.iloc[:, 0]
        )
        seg_comp_df.loc[matched_dfa_only.index.values.tolist(), col_name_both_filt] = (
            matched_dfa_only.iloc[:,0]
        )

        # mutual matches with seg a filtered only
        dfa = seg_comp_df[
            (seg_comp_df["source"] == self.data_names[0])
            & (seg_comp_df["low_quality_cells"] == False)
        ]
        dfb = seg_comp_df[seg_comp_df["source"] == self.data_names[1]]
        col_name_afilt = (
            "match_" + self.data_names[0] + "_filt_" + self.data_names[1] + "_unfilt"
        )
        # dfa filt only
        matched_dfa_only, matched_dfb_only = get_mutual_matches(dfa, dfb, nn_dist)
        seg_comp_df.loc[matched_dfb_only.index.values.tolist(), col_name_afilt] = (
            matched_dfb_only.iloc[:,0]
        )
        seg_comp_df.loc[matched_dfa_only.index.values.tolist(), col_name_afilt] = (
            matched_dfa_only.iloc[:,0]
        )

        # mutual matches with seg b filtered only
        dfa = seg_comp_df[seg_comp_df["source"] == self.data_names[0]]
        dfb = seg_comp_df[
            (seg_comp_df["source"] == self.data_names[1])
            & (seg_comp_df["low_quality_cells"] == False)
        ]
        col_name_bfilt = (
            "match_" + self.data_names[0] + "_unfilt_" + self.data_names[1] + "_filt"
        )
        # dfa filt only
        matched_dfa_only, matched_dfb_only = get_mutual_matches(dfa, dfb, nn_dist)
        seg_comp_df.loc[matched_dfb_only.index.values.tolist(), col_name_bfilt] = (
            matched_dfb_only.iloc[:,0]
        )
        seg_comp_df.loc[matched_dfa_only.index.values.tolist(), col_name_bfilt] = (
            matched_dfa_only.iloc[:,0]
        )

        # save results
        if save:
            result_path = f"{savepath}{bc}_seg_comp_df_{self.data_names[0]}_and_{self.data_names[1]}_populated.csv"
            seg_comp_df.to_csv(result_path, index=True)
            print("Saved to: " + result_path)
        return seg_comp_df

    def generate_sankey_diagram(
        self,
        seg_comp_df,
        bc,
        save=True,
        savepath="/allen/programs/celltypes/workgroups/hct/emilyg/",
    ):
        """
        Creates sankey diagram comparing two segmentation results using segmentation comparison dataframe created by collect_mutual_match_and_doublets function
        INPUTS
            seg_comp_df: dataframe with unique index describing cell spatial locations (x and y), segmentation identifier, low quality cell identifier, mutual matches, and putative doublets.
            bc: unique identifier for section, used for creating the save name for the dataframe, string.
            save: whether to save the final results. Bool, default true
            savepath: path to which results should be saved, string
        OUTPUTS
            fig: sankey diagram figure
        """
        nodes_df, unknown_unmatched_cells = create_node_df_sankey(
            seg_comp_df, bc, save, savepath
        )
        links_df = create_link_df_sankey(nodes_df, bc, save, savepath)
        # Sankey plot setup
        data_trace = dict(
            type="sankey",
            node=dict(
                line=dict(color="black", width=0),
                label=nodes_df["Label"].dropna(axis=0, how="any"),
                color=nodes_df["Color"],
            ),
            link=dict(
                source=links_df["Source"].dropna(axis=0, how="any"),
                target=links_df["Target"].dropna(axis=0, how="any"),
                value=links_df["Value"].dropna(axis=0, how="any"),
                color=links_df["Link Color"].dropna(axis=0, how="any"),
            ),
        )

        layout = dict(
            title="Cell segmentation comparison human "
            + bc
            + " "
            + self.data_names[0]
            + " vs "
            + self.data_names[1],
            height=772,
            font=dict(size=10),
        )
        fig = go.Figure(dict(data=[data_trace], layout=layout))
        # save interactive file
        if save:
            filepath = (
                savepath
                + "/segmentation_comparison_"
                + bc
                + "_"
                + self.data_names[0]
                + "_"
                + self.data_names[1]
                + ".png"
            )
            fig.write_image(filepath)
        fig.show()
        return fig, unknown_unmatched_cells, filepath

    def scaling_check(self, seg_comp_df):
        """
        View two sections plotted on same axes, to check for scaling issues
        INPUTS
            seg_comp_df:dataframe with unique index describing cell spatial locations (x and y), segmentation identifier, low quality cell identifier, mutual matches, and putative doublets.
        """
        # overview of both sections plotted on same axes
        fig, ax = plt.subplots()
        filtered_seg_a_df = seg_comp_df[
            (seg_comp_df["source"] == self.data_names[0])
            & (seg_comp_df["low_quality_cells"] == False)
        ]
        filtered_seg_a_df.plot(
            "center_x",
            "center_y",
            kind="scatter",
            s=0.1,
            label="high quality cells " + self.data_names[0],
            ax=ax,
            color="blue",
            alpha=0.5,
        )
        filtered_seg_b_df = seg_comp_df[
            (seg_comp_df["source"] == self.data_names[1])
            & (seg_comp_df["low_quality_cells"] == False)
        ]
        filtered_seg_b_df.plot(
            "center_x",
            "center_y",
            kind="scatter",
            s=0.3,
            label="high quality cells " + self.data_names[1],
            ax=ax,
            color="red",
            alpha=0.5,
        )
        plt.title("Both segmentations, overlaid")
        plt.axis("equal")

        # find center of one section to zoom in on
        arr = filtered_seg_a_df[["center_x", "center_y"]].values
        center = [np.mean(arr[:, 0]), np.mean(arr[:, 1])]
        x_range = [center[0] - center[0] * 0.05, center[0] + center[0] * 0.05]
        y_range = [center[1] - center[1] * 0.05, center[1] + center[1] * 0.05]

        # plot same area for both sections
        fig, ax = plt.subplots()
        filtered_seg_a_df = seg_comp_df[
            (seg_comp_df["source"] == self.data_names[0])
            & (seg_comp_df["low_quality_cells"] == False)
        ]
        subs_a = filtered_seg_a_df[
            (filtered_seg_a_df["center_x"].between(x_range[0], x_range[1]))
            & (filtered_seg_a_df["center_y"].between(y_range[0], y_range[1]))
        ]
        subs_a.plot(
            "center_x",
            "center_y",
            kind="scatter",
            s=2,
            label=self.data_names[0],
            ax=ax,
            color="blue",
            alpha=0.5,
        )
        plt.title("Both segmentations, overlaid")
        filtered_seg_b_df = seg_comp_df[
            (seg_comp_df["source"] == self.data_names[1])
            & (seg_comp_df["low_quality_cells"] == False)
        ]
        subs_b = filtered_seg_b_df[
            (filtered_seg_b_df["center_x"].between(x_range[0], x_range[1]))
            & (filtered_seg_b_df["center_y"].between(y_range[0], y_range[1]))
        ]
        subs_b.plot(
            "center_x",
            "center_y",
            kind="scatter",
            s=10,
            label=self.data_names[1],
            ax=ax,
            facecolors="none",
            edgecolors="r",
            alpha=0.5,
        )
        plt.title("Both segmentations, overlaid, zoomed in to center")
        plt.axis("equal")
        return


def filter_and_cluster_twice(
    input_ad,
    n_hvgs=2000,
    min_max_counts=3,
    min_transcript_counts=50,
    n_pcs=[50, 20],
    n_iterations=[1, 1],
    plot_stuff=True,
    run_preprocessing=False,
):
    """
    filter genes and cells, then run 2 rounds of Leiden clustering on an anndata object
    resulting clusters are named "leiden_XX_YY" where "XX" and "YY" are the numbers of the first and second
    round clusters

    this function returns a modified copy of the anndata object with potentially fewer genes and cells!
    """

    # switch to array instead of sparse matrix:
    if isinstance(input_ad.X, sparse.csr.csr_matrix):
        input_ad.X = input_ad.X.toarray()

    print("converted to array  ")
    if "gene" in input_ad.var.columns:
        low_detection_genes = input_ad.var.iloc[
            np.nonzero(np.max(input_ad.X, axis=0) <= min_max_counts)
        ].gene
    else:
        low_detection_genes = input_ad.var.iloc[
            np.nonzero(np.max(input_ad.X, axis=0) <= min_max_counts)
        ].index

    # throw out genes if no cells have more than 3 counts
    if "gene" in input_ad.var.columns:
        gene_list = [g for g in input_ad.var.gene if g not in low_detection_genes]
    else:
        gene_list = [g for g in input_ad.var.index if g not in low_detection_genes]

    to_cluster = input_ad[input_ad.obs.transcript_counts >= min_transcript_counts, :]

    # this is to prevent multiple "leiden" columns from being added to the obs dataframe
    to_cluster.obs = to_cluster.obs.loc[
        :,
        [
            c
            for c in to_cluster.obs.columns
            if c not in ["leiden", "leiden_0", "leiden_1"]
        ],
    ]

    if run_preprocessing:
        # Normalizing to median total counts
        print("normalizing total counts")
        sc.pp.normalize_total(to_cluster)
        # Logarithmize the data
        sc.pp.log1p(to_cluster)

        sc.pp.highly_variable_genes(to_cluster, n_top_genes=n_hvgs)

    sc.tl.pca(to_cluster, n_comps=n_pcs[0])

    sc.pp.neighbors(to_cluster)

    if plot_stuff:
        sc.tl.umap(to_cluster)
        sc.pl.pca_variance_ratio(to_cluster, n_pcs[0], log=True)

    sc.tl.leiden(to_cluster, n_iterations=n_iterations[0])
    to_cluster.obs.rename(columns={"leiden": "leiden_0"}, inplace=True)

    if plot_stuff:
        sc.pl.umap(to_cluster, color=["leiden_0"], size=2)

    # per cluster, repeat PCA and clustering...
    # this duplicates the data! but it's necessary because the scanpy functions would create a copy of the subset anyway(?)
    print("2nd round of clustering")
    all_subs = []
    for cl in to_cluster.obs.leiden_0.unique():
        subcopy = to_cluster[to_cluster.obs.leiden_0 == cl, :].copy()
        sc.tl.pca(subcopy, n_comps=n_pcs[1])
        sc.pp.neighbors(subcopy)
        sc.tl.leiden(subcopy, n_iterations=n_iterations[1])
        if plot_stuff:
            sc.pl.umap(subcopy, color=["leiden"])
        subcopy.obs["leiden_1"] = [
            "leiden_" + str(cl).zfill(2) + "_" + str(g).zfill(2)
            for g in subcopy.obs.leiden
        ]

        all_subs.append(subcopy)
    print("concatenating")
    iterative_clusters_ad = ad.concat(all_subs)

    return iterative_clusters_ad


def detection_ratio_plots(
    gene_ratio_df, data_names=DEFAULT_DATA_NAMES, figsize=[15, 15]
):

    sorted_genes = [
        str(s) for s in gene_ratio_df.mean(axis=1).sort_values().index.values
    ]

    plt.figure(figsize=figsize)
    plt.subplot(3, 1, 1)
    p = sns.boxplot(
        gene_ratio_df.loc[sorted_genes, :].T,
    )
    p.set_yscale("log")
    p.set_xlabel("gene", fontsize=20)
    p.set_ylabel(
        "detection ratio\n" + data_names[1] + " / " + data_names[0], fontsize=20
    )
    ax = plt.gca()
    ax.tick_params(axis="x", labelrotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=20, which="major")
    ax.tick_params(axis="y", labelsize=10, which="minor")

    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    plt.plot(ax.get_xlim(), [1, 1], "-")

    # sort columns for boxplot based on mean detection ratio:
    sorted_categories = gene_ratio_df.columns[np.argsort(gene_ratio_df.mean(axis=0))]

    plt.subplot(3, 1, 2)
    ax = sns.boxplot(gene_ratio_df.loc[:, sorted_categories])
    plt.ylabel("transcript detection ratio " + data_names[1] + " / " + data_names[0])
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    plt.plot(ax.get_xlim(), [1, 1])

    plt.subplot(3, 1, 3)
    h = plt.hist(np.log10(gene_ratio_df.values.ravel()), bins=100)
    median_ratio = np.nanmedian(gene_ratio_df.values.ravel())
    mean_ratio = np.nanmean(gene_ratio_df.values.ravel())
    plt.plot(
        np.log10(median_ratio) * np.ones([2]),
        [0, 1.1 * h[0].max()],
        label="median ratio "
        + data_names[1]
        + " / "
        + data_names[0]
        + " : "
        + str(median_ratio)[:5],
    )
    plt.plot(
        np.log10(mean_ratio) * np.ones([2]),
        [0, 1.1 * h[0].max()],
        label="mean ratio "
        + data_names[1]
        + " / "
        + data_names[0]
        + " : "
        + str(mean_ratio)[:5],
    )
    plt.legend()
    plt.title(
        "ratio of cell segmented counts for shared genes\n "
        + data_names[1]
        + " / "
        + data_names[0],
        fontsize=20,
    )
    plt.xlabel("log10(counts ratio)", fontsize=20)
    ax = plt.gca()
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()


def generate_label_confusion(
    input_ad,
    column_pairs=[
        ["iterative_subclass", "subclass_name"],
        ["iterative_leiden", "supertype_name"],
        ["iterative_leiden", "subclass_name"],
    ],
):

    # identifies fraction correctly matching labels in pairs of columns supplied by user in `column_pairs`

    column_pair_results = []
    for column_pair in column_pairs:
        row_dfs = []
        for g in input_ad.obs[column_pairs[0][0]].unique():
            g_df = (
                input_ad.obs.loc[input_ad.obs[column_pairs[0][0]] == g, column_pairs[0]]
                .groupby(column_pairs[0][1], observed=False)
                .count()
            )
            g_df.rename(
                columns={column_pairs[0][0]: column_pairs[0][0] + "_" + g}, inplace=True
            )
            row_dfs.append(g_df)
        tdf = pd.concat(row_dfs, axis=1)
        tdf_norm = tdf.copy().astype(float)
        for ii in tdf_norm.index:
            tdf_norm.loc[ii, :] = tdf_norm.loc[ii, :] / tdf_norm.loc[ii, :].sum()

        column_pair_results.append(tdf_norm)

    return {
        c_p[0] + "_" + c_p[1]: column_pair_results[ii]
        for ii, c_p in enumerate(column_pairs)
    }


def find_best_match_groups(
    ad0, ad1, group_names=["leiden_1", "leiden_1"], in_place=True
):

    g_o_m_0 = grouped_obs_mean(ad0, group_names[0])
    g_o_m_1 = grouped_obs_mean(ad1, group_names[1])

    # these can have different numbers of clusters and different sets of observed genes.
    # subset to shared genes for this and deal with non-square correlation matrix

    shared_genes = list(g_o_m_0.loc[g_o_m_0.index.isin(g_o_m_1.index), :].index)
    correlation_array = np.corrcoef(
        pd.concat([g_o_m_0.loc[shared_genes, :], g_o_m_1.loc[shared_genes, :]], axis=1),
        rowvar=0,
    )
    shape0 = g_o_m_0.loc[shared_genes, :].shape
    shape1 = g_o_m_1.loc[shared_genes, :].shape
    array_01 = correlation_array[: shape0[1], shape0[1] :]
    array_10 = array_01.T

    match_01 = {r: np.argmax(array_01[r, :]) for r in range(array_01.shape[0])}
    match_10 = {r: np.argmax(array_10[r, :]) for r in range(array_10.shape[0])}

    # find mutual matches:
    mutual_matches = {}
    for k in match_01:
        if match_10[match_01[k]] == k:
            mutual_matches.update({k: match_01[k]})

    if in_place:
        ad0.obs["matched_" + group_names[0]] = ""
        ad1.obs["matched_" + group_names[0]] = ""

        for mm in mutual_matches:

            match_mask0 = ad0.obs[group_names[0]] == g_o_m_0.columns[mm]
            ad0.obs.loc[match_mask0, ["matched_" + group_names[0]]] = (
                "matched_" + group_names[0] + "_" + str(mm)
            )

            match_mask1 = ad1.obs[group_names[0]] == g_o_m_1.columns[mutual_matches[mm]]
            ad1.obs.loc[match_mask1, ["matched_" + group_names[0]]] = (
                "matched_" + group_names[0] + "_" + str(mm)
            )
    else:
        ad0_out = ad0.copy()
        ad1_out = ad1.copy()

        ad0_out.obs["matched_" + group_names[0]] = ""
        ad1_out.obs["matched_" + group_names[0]] = ""

        for mm in mutual_matches:

            match_mask0 = ad0_out.obs[group_names[0]] == g_o_m_0.columns[mm]
            ad0_out.obs.loc[match_mask0, ["matched_" + group_names[0]]] = (
                "matched_" + group_names[0] + "_" + str(mm)
            )

            match_mask1 = (
                ad1_out.obs[group_names[0]] == g_o_m_1.columns[mutual_matches[mm]]
            )
            ad1_out.obs.loc[match_mask1, ["matched_" + group_names[0]]] = (
                "matched_" + group_names[0] + "_" + str(mm)
            )
        return (ad0, ad1)


def get_column_ordering(df, ordered_rows):

    # function to get sorted columns matched to input ordered rows
    # runs through each entry in `ordered_rows` and adds first non-repeated column from the ranked list to that category
    # repeats until all the columns are done
    # ranking is based on the values in the array, in this case it is the fraction of cells that mapped to each input row
    flat_ordering = []
    empty_columns = {r: [] for r in ordered_rows}
    while len(flat_ordering) < df.shape[1]:
        for r in ordered_rows:
            r_columns = [
                df.columns[rr]
                for rr in df.loc[r, :].argsort()[::-1]
                if not (df.columns[rr] in flat_ordering)
            ]
            if len(r_columns) > 0 and len(flat_ordering) < df.shape[1]:
                empty_columns[r].append(r_columns[0])
                flat_ordering.append(r_columns[0])

    output = []
    [output.extend(empty_columns[k]) for k in empty_columns]
    return output


def get_segmentation_data(
    bc,
    anndata_a,
    anndata_b,
    seg_name_a,
    seg_name_b,
    save,
    savepath,
    reuse_saved,
    min_transcripts,
):
    """
    Loads segmentation data from anndata object
    Inputs:
        bc: unique section identifier, string
        seg_name_a, seg_name_b: names of segmentations to be compared, string
        anndata_path_a, anndata_path_b: path to segmented anndata objects, string
        save: whether to save the intermediary dataframe (useful if running functions individually), bool
        savepath: path to which data should be saved. Defaults to seg_b_path if not specified.
        min_transcripts: minimum number of transcripts needed to define a cell too low quality to be considered for mapping
    RETURNS
        seg_comp_df: dataframe describing cell spatial locations (x and y), segmentation identifier, low quality cell identifier, with unique index
    """
    savepath = (
        savepath + bc + "_seg_comp_df_" + seg_name_a + "_and_" + seg_name_b + ".csv"
    )
    if Path(savepath).exists() and reuse_saved:
        seg_comp_df = pd.read_csv(savepath, index_col=0)
    else:
        seg_dfs = []
        for i in range(2):
            if i == 0:
                seg_h5ad = anndata_a
                name = seg_name_a
            else:
                seg_h5ad = anndata_b
                name = seg_name_b
            df_cols = ["center_x", "center_y", "source", "low_quality_cells"]
            # sum across .X to get sum transcripts/cell > 40
            if "low_quality_cells" not in seg_h5ad.obs.columns.tolist():
                high_quality_cells = [
                    idx
                    for idx, x in enumerate(seg_h5ad.X.sum(axis=1))
                    if x >= min_transcripts
                ]
                seg_h5ad.obs["low_quality_cells"] = [True] * len(seg_h5ad.obs)
                seg_h5ad.obs.iloc[high_quality_cells, -1] = False
            seg_h5ad.obs.loc[:, "source"] = name
            # save only necessary columns
            if "center_x" not in seg_h5ad.obs.columns.tolist():
                seg_h5ad.obs["center_x"] = seg_h5ad.obsm["spatial"][:, 0]
                seg_h5ad.obs["center_y"] = seg_h5ad.obsm["spatial"][:, 1]
            seg_df = seg_h5ad.obs[df_cols]
            seg_df.index = f"{name}_" + seg_df.index
            seg_dfs.append(seg_df)
        seg_comp_df = pd.concat(seg_dfs, axis=0)
        seg_comp_df.index = seg_comp_df.index.astype(str)
    if save:
        seg_comp_df.to_csv(savepath)
    return seg_comp_df


def get_mutual_matches(dfa, dfb, nn_dist):
    """
    Compare x and y locations using nearest neighbor function of each segmentation result to identify same cells across segmentations
    INPUTS
        dfa, dfb: subset dataframe for each segmentation
        nn_dist: distance cutoff for identifying likely same cell between segmentations. float.
    OUTPUT:
        mutual_match: Dictionary of cell IDs to matched cell ids. Keys are stacked source a and source b cell ids, with values being match from other source (so source a cell id keys have source b cell id values, and vice versa)
    """

    # get single nearest neighbor for each
    tree_a = cKDTree(dfa[["center_x", "center_y"]].copy())
    dists_a, inds_a = tree_a.query(
        dfb[["center_x", "center_y"]].copy(), 1
    )  # gives me index locs for dfa with neighbor dfb
    tree_b = cKDTree(dfb[["center_x", "center_y"]].copy())
    dists_b, inds_b = tree_b.query(
        dfa[["center_x", "center_y"]].copy(), 1
    )  # vice versa
    # get mutually matching pairs and save (index of seg_comp_df is str)
    match_to_dfb = pd.DataFrame(
        data=dfa.iloc[inds_a].index.values.tolist(),
        index=pd.Index(dfb.index, name="match_dfb_index"),
        columns=["match_dfa_index"],
    )
    match_to_dfb["same_cell"] = np.where(dists_a <= nn_dist, True, False)
    matched_dfb_only = match_to_dfb[match_to_dfb["same_cell"] == True]

    match_to_dfa = pd.DataFrame(
        data=dfb.iloc[inds_b].index.values.tolist(),
        index=pd.Index(dfa.index, name="match_dfa_index"),
        columns=["match_dfb_index"],
    )
    match_to_dfa["same_cell"] = np.where(dists_b <= nn_dist, True, False)
    matched_dfa_only = match_to_dfa[match_to_dfa["same_cell"] == True]
    return matched_dfb_only, matched_dfa_only


def create_node_df_sankey(
    seg_comp_df,
    barcode,
    save=True,
    savepath="/allen/programs/celltypes/workgroups/hct/emilyg/reseg_project/new_seg/",
):
    """
    Dataframe describing each node to be included in sankey diagram. Additionally contains data helpful for generating required links dataframe.
    INPUTS
        seg_comp_df: dataframe with unique index describing cell spatial locations (x and y), segmentation identifier, low quality cell identifier, mutual matches, and putative doublets.
        barcode: unique identifier for section, used for creating the save name for the dataframe, string.
        save: whether to save the final results. Bool, default true
        savepath: path to which results should be saved, string
    OUTPUTS
        nodes_df: dataframe containing node label, color, and value for sankey diagram creation, with additional level and source keys for ease of link dataframe creation
    """

    color_dict = {
        seg_comp_df.source.unique().tolist()[0]: "red",
        seg_comp_df.source.unique().tolist()[1]: "blue",
    }
    nodes_df = pd.DataFrame(columns=["Label", "Color", "Level", "Source", "Value"])
    unknown_unmatched_cells = {}
    for source, g in seg_comp_df.groupby("source"):
        new_rows = []
        low_q_and_match = g.groupby("low_quality_cells").agg(
            "count"
        )  # gives me low q t/f counts, and matched.
        total_cells = low_q_and_match.iloc[:, 0].sum()
        nodes_df.loc[len(nodes_df)] = {
            "Label": "total <br>" + str(total_cells),
            "Color": color_dict[source],
            "Level": 0,
            "Source": source,
            "Value": total_cells,
        }
        # low and normal quality cells
        nodes_df.loc[len(nodes_df)] = {
            "Label": "low quality cells <br>" + str(low_q_and_match.iloc[1, 1]),
            "Color": color_dict[source],
            "Level": 1,
            "Source": source,
            "Value": low_q_and_match.iloc[1, 1],
        }
        nodes_df.loc[len(nodes_df)] = {
            "Label": "normal quality cells <br>" + str(low_q_and_match.iloc[0, 1]),
            "Color": color_dict[source],
            "Level": 1,
            "Source": source,
            "Value": low_q_and_match.iloc[0, 1],
        }
        # matched and unmatched cells
        # because row 1 in agg is false only! so false x low (normal) and false x match (unmatch!)
        matched_cells = low_q_and_match.iloc[0, 1] - low_q_and_match.iloc[0, 3]
        nodes_df.loc[len(nodes_df)] = {
            "Label": "matched cells <br>" + str(matched_cells),
            "Color": color_dict[source],
            "Level": 2,
            "Source": source,
            "Value": matched_cells,
        }
        nodes_df.loc[len(nodes_df)] = {
            "Label": "unmatched cells <br>" + str(low_q_and_match.iloc[0, 3]),
            "Color": color_dict[source],
            "Level": 2,
            "Source": source,
            "Value": low_q_and_match.iloc[0, 3],
        }
        # raise flag if too few cells matched, may indicate scaling issue
        if matched_cells <= (0.1 * len(g)):
            raise ValueError(
                "The number of matched cells is less than 1% of total cells present. Please check your inputs for potential scaling issues."
            )

        rem_col_names = [
            x for x in g.columns[5:] if source + "_unfilt" not in x
        ]  # get remaining columns
        # pulling this to access unknown unmatched cells easily later
        high_q_df = g[g[g.columns.tolist()[3]] == False]
        unmatched_df = high_q_df[high_q_df[g.columns.tolist()[4]].isna() == False]
        rem_col_df = unmatched_df.loc[:, rem_col_names]
        rem_col_counts = rem_col_df.isna().value_counts().reset_index()
        name = " ".join(rem_col_names[0].split("_")) + "<br>"
        fcounts = rem_col_counts[rem_col_counts[rem_col_names[0]] == False][
            "count"
        ].values.tolist()[0]
        nodes_df.loc[len(nodes_df)] = {
            "Label": name + str(fcounts),
            "Color": color_dict[source],
            "Level": 3,
            "Source": source,
            "Value": fcounts,
        }
        # remaining unmatched cells (known or unknown)
        if len(rem_col_counts) > 1:
            rem_unmatched_cells = rem_col_counts[
                rem_col_counts[rem_col_names[0]] == True
            ]["count"].values.tolist()[0]
            unknown_unmatched_cells[source] = rem_col_df[
                rem_col_df[rem_col_names[0]].isna() == True
            ].index.values.tolist()
            name = " ".join(rem_col_names[-1].split("_"))
            nodes_df.loc[len(nodes_df)] = {
                "Label": name + str(rem_unmatched_cells),
                "Color": color_dict[source],
                "Level": 3,
                "Source": source,
                "Value": rem_unmatched_cells,
            }
    if save:
        nodes_df.to_csv(savepath + "/sankey_nodes_df" + barcode + ".csv")
    return nodes_df, unknown_unmatched_cells


def create_link_df_sankey(
    nodes_df,
    barcode,
    save=True,
    savepath="/allen/programs/celltypes/workgroups/hct/emilyg/reseg_project/new_seg/",
):
    """
    Generates links dataframe from nodes dataframe, connecting relevant nodes to one another and ensuring distinct segmentations are separated.
    INPUTS
        nodes_df: dataframe containing information on each node for sankey diagram.
        barcode: unique identifier for section, used for creating the save name for the dataframe, string.
        save: whether to save the final results. Bool, default true
        savepath: path to which results should be saved, string
    OUTPUTS
        links_df: dataframe containing source, target, value, and colors for sankey diagram creation
    """

    link_color_dict = {"red": "rgb(205, 209, 228)", "blue": "rgb(205, 209, 228)"}
    links_df = pd.DataFrame(columns=["Source", "Target", "Value", "Link Color"])
    for i, row in nodes_df.iterrows():
        if "total" in row.Label:
            continue  # skip 0 since all comes from zero
        else:
            target = i
            value = row["Value"]
            source = nodes_df[
                (nodes_df["Level"] == row["Level"] - 1)
                & (nodes_df["Source"] == row["Source"])
            ].index.values.tolist()
            if len(source) > 1 and row["Level"] != 3:
                source = [
                    i
                    for i, x in nodes_df.iloc[source, :].iterrows()
                    if "un" not in x["Label"]
                    if "low" not in x["Label"]
                ][0]
            elif len(source) > 1 and row["Level"] == 3:
                source = [
                    i
                    for i, x in nodes_df.iloc[source, :].iterrows()
                    if "un" in x["Label"]
                ][0]
            else:
                source = source[0]
            prev_row = nodes_df.iloc[source, :]
            link_color = link_color_dict[prev_row["Color"]]
            links_df.loc[len(links_df)] = {
                "Source": source,
                "Target": target,
                "Value": value,
                "Link Color": link_color,
            }

    # special case: matched cells should have same name, new source, new color
    matched_cell_rows = nodes_df[
        (nodes_df["Label"].str.contains("matched cells"))
        & (~nodes_df["Label"].str.contains("un"))
    ]
    matched_cell_val = matched_cell_rows["Value"].values.tolist()[0]
    nodes_df.loc[len(nodes_df)] = {
        "Label": "matched cells <br> " + str(matched_cell_val),
        "Color": "violet",
        "Level": matched_cell_rows["Level"].values.tolist()[0],
        "Source": "Both",
        "Value": matched_cell_val,
    }
    # find matched col for both in nodes_df, then find rows with those IDS in links df and connect them
    update_idxs = links_df[
        links_df["Target"].isin(matched_cell_rows.index.values.tolist())
    ].index.values.tolist()
    links_df.loc[update_idxs, "Target"] = [
        nodes_df[nodes_df["Source"] == "Both"].index.values
    ] * len(update_idxs)
    if save:
        links_df.to_csv(savepath + "/sankey_links_df.csv")
    return links_df
