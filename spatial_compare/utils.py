import anndata as ad
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from sklearn.metrics import adjusted_rand_score


def grouped_obs_mean(adata, group_key, layer=None, gene_symbols=None):
    """
    Calculate the mean expression of observations grouped by a specified key.
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    group_key : str
        Key in `adata.obs` to group by.
    layer : str, optional
        Layer in `adata` to use for expression values. If None, use `adata.X`.
    gene_symbols : str, optional
        Column in `adata.var` to use as gene symbols. If None, use `adata.var_names`.
    Returns
    -------
    pd.DataFrame
        DataFrame with mean expression values for each group. Columns correspond to groups,
        and rows correspond to genes.
    """

    if layer is not None:
        getX = lambda x: x.layers[layer]
    else:
        getX = lambda x: x.X
    if gene_symbols is not None:
        new_idx = adata.var[gene_symbols]
    else:
        new_idx = adata.var_names

    grouped = adata.obs.groupby(group_key)
    out = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=new_idx,
    )

    for group, idx in grouped.indices.items():
        X = getX(adata[idx])
        out[group] = np.ravel(X.mean(axis=0, dtype=np.float64)).tolist()

    return out


def spatial_detection_scores(
    reference: pd.DataFrame,
    query: pd.DataFrame,
    plot_stuff=True,
    query_name: str = "query data",
    comparison_column="transcript_counts",
    category="supercluster_name",
    n_bins=50,
    in_place=True,
    non_spatial=False,
):
    """
    Calculate and plot spatial detection scores for query data compared to reference data.

    Parameters:
        reference (pd.DataFrame): The reference data.
        query (pd.DataFrame): The query data.
        plot_stuff (bool, optional): Whether to plot the results. Defaults to True.
        query_name (str, optional): The name of the query data. Defaults to "query data".
        category (str, optional): The category column to compare. Defaults to "supercluster_name".
        n_bins (int, optional): The number of bins for spatial grouping. Defaults to 50.
        in_place (bool, optional): Whether to modify the query data in place. Defaults to True.
        non_spatial (bool, optional): Whether to compare to an ungrouped mean/std. Defaults to False.

    Returns:
        dict: A dictionary containing the bin image, extent, query data, and reference data (if in_place is False).
    """
    # code goes here

    if category not in reference.columns or category not in query.columns:
        raise ValueError("category " + category + " not in reference and query inputs")

    shared_category_values = list(
        set(reference[category].unique()) & set(query[category].unique())
    )
    if (
        len(shared_category_values) < query[category].unique().shape[0]
        or len(shared_category_values) < reference[category].unique().shape[0]
    ):
        in_place = False

    if in_place:
        s2 = query.loc[query[category].isin(shared_category_values), :]
        s1 = reference.loc[reference[category].isin(shared_category_values), :]
    else:
        s2 = query.loc[query[category].isin(shared_category_values), :].copy()
        s1 = reference.loc[reference[category].isin(shared_category_values), :].copy()

    means = s1.groupby(category, observed=True)[comparison_column].mean()
    stds = s1.groupby(category, observed=True)[comparison_column].std()

    # if you want to compare to an ungrouped mean/std, try this:
    if non_spatial:
        means[:] = means.mean()
        stds[:] = stds.mean()

    s2["detection_relative_z_score"] = 0.0
    s2["detection_difference"] = 0.0
    s2["detection_ratio"] = 0.0

    for c, gb in s2.groupby(category, observed=True):
        if c not in shared_category_values:
            continue

        s2.loc[s2[category] == c, ["detection_relative_z_score"]] = (
            (s2.loc[s2[category] == c, [comparison_column]] - means[c]) / stds[c]
        ).values
        s2.loc[s2[category] == c, ["detection_difference"]] = (
            s2.loc[s2[category] == c, [comparison_column]] - means[c]
        ).values
        s2.loc[s2[category] == c, ["log_10_detection_ratio"]] = np.log10(
            (s2.loc[s2[category] == c, [comparison_column]] / means[c]).values
        )

    s2["xy_bucket"] = list(
        zip(
            pd.cut(s2.x_centroid, n_bins, labels=list(range(n_bins))),
            pd.cut(s2.y_centroid, n_bins, labels=list(range(n_bins))),
        )
    )

    binx = s2.groupby("xy_bucket").x_centroid.mean()
    biny = s2.groupby("xy_bucket").y_centroid.mean()

    z_score = s2.groupby("xy_bucket").detection_relative_z_score.mean()
    difference = s2.groupby("xy_bucket").detection_difference.mean()
    log_ratio = s2.groupby("xy_bucket").log_10_detection_ratio.mean()
    n_cells = s2.groupby("xy_bucket").x_centroid.count()

    bin_image_z_score = np.zeros([n_bins, n_bins])
    bin_image_difference = np.zeros([n_bins, n_bins])
    bin_image_ratio = np.zeros([n_bins, n_bins])
    bin_image_counts = np.zeros([n_bins, n_bins])

    extent = [np.min(binx), np.max(binx), np.min(biny), np.max(biny)]
    for coord in binx.index:
        bin_image_z_score[coord[1], coord[0]] = z_score[coord]
        bin_image_difference[coord[1], coord[0]] = difference[coord]
        bin_image_ratio[coord[1], coord[0]] = log_ratio[coord]
        bin_image_counts[coord[1], coord[0]] = n_cells[coord]

    if plot_stuff:
        if non_spatial:
            title_string = "Non-spatial Detection Scores"
        else:
            title_string = "Spatial Detection Scores"
        min_maxes = {
            "detection z-score": [bin_image_z_score, [-1, 1]],
            "total counts difference": [bin_image_difference, [-100, 100]],
            "log10(detection ratio)": [bin_image_ratio, [-1, 1]],
        }

        fig, axs = plt.subplots(1, 3, figsize=[15, 5])
        if non_spatial:
            fig.suptitle(title_string + "\n" + query_name)
        else:
            fig.suptitle(
                title_string
                + "\n"
                + query_name
                + " grouped by "
                + category
                + " and spatially binned"
            )
        for ii, plot_name in enumerate(min_maxes.keys()):
            ax = axs[ii]
            pcm = ax.imshow(
                min_maxes[plot_name][0],
                extent=extent,
                cmap="coolwarm_r",
                vmin=min_maxes[plot_name][1][0],
                vmax=min_maxes[plot_name][1][1],
            )
            fig.colorbar(pcm, ax=ax, shrink=0.7)
            ax.set_title(query_name + "\n" + plot_name)

    if in_place:

        return dict(
            z_score_image=bin_image_z_score,
            difference_image=bin_image_difference,
            ratio_image=bin_image_ratio,
            extent=extent,
            count_image=bin_image_counts,
            query=True,
            reference=True,
        )

    else:
        return dict(
            z_score_image=bin_image_z_score,
            difference_image=bin_image_difference,
            ratio_image=bin_image_ratio,
            extent=extent,
            count_image=bin_image_counts,
            query=s2,
            reference=s1,
        )


def summarize_and_plot(
    spatial_density_results,
    min_cells_per_bin=50,
    z_score_limit=-0.5,
    area_frac_limit=0.1,
    figsize=[15, 15],
    plot_columns=9,
    plot_stuff=True,
    title_mapping={},
):
    """
    Summarizes and plots spatial density results.
    Parameters:
    - spatial_density_results (dict): A dictionary containing spatial density results.
    - min_cells_per_bin (int): The minimum number of cells per bin. Default is 50.
    - z_score_limit (float): The z-score limit. Default is -0.5.
    - area_frac_limit (float): The area fraction limit. Default is 0.1.
    - figsize (list): The figure size. Default is [15, 15].
    - plot_columns (int): The number of plot columns in figure subplot. Default is 9.
    - plot_stuff (bool): Whether to plot the results. Default is True.
    - title_mapping (dict): A dictionary mapping spatial_density_results keys to titles for the plots. Default is an empty dictionary.
    Returns:
    - results (list): A list of dictionaries containing the summarized results. Each dictionary contains the following keys:
        - key (float): same as original input key
        - bad_frac (float): The fraction of tissue bins with z-scores below the z-score limit.
        - area_frac_limit (float): The area fraction limit.
        - failed (bool): Whether the fraction of tissue bins with z-scores below the z-score limit is greater than or equal to the area fraction limit.
        - mean (float): The mean of the non-zero z-scores.
        - stdev (float): The standard deviation of the non-zero z-scores.
    """

    fails_vs_rnaseq = []
    results = []
    if plot_stuff:
        plt.figure(figsize=figsize)

    for ii, z in enumerate(sorted(list(spatial_density_results.keys()))):
        # estimate tissue bins:
        tissue_bins = spatial_density_results[z]["count_image"] > min_cells_per_bin
        bad_frac = np.sum(
            spatial_density_results[z]["z_score_image"][tissue_bins] <= z_score_limit
        ) / np.sum(tissue_bins)
        image_to_show = spatial_density_results[z]["z_score_image"].copy()
        image_to_show[np.logical_not(tissue_bins)] = 0

        if plot_stuff:
            if z in title_mapping:
                title_prefix = title_mapping[z]
            else:
                title_prefix = ""

            plt.subplot(
                1 + len(list(spatial_density_results.keys())) // plot_columns,
                plot_columns,
                ii + 1,
            )

            ax = plt.imshow(
                image_to_show,
                vmin=-1,
                vmax=1,
                extent=spatial_density_results[z]["extent"],
                cmap="coolwarm_r",
            )

            if bad_frac >= area_frac_limit:
                plt.title(title_prefix + " Fail", fontdict={"size": 12})
            else:
                plt.title(title_prefix, fontdict={"size": 12})
            plt.xticks([])
            plt.yticks([])

        if bad_frac >= area_frac_limit:
            fails_vs_rnaseq.append(z)

        results.append(
            dict(
                key=z,
                bad_frac=bad_frac,
                area_frac_limit=area_frac_limit,
                failed=bad_frac >= area_frac_limit,
                mean=np.mean(image_to_show[image_to_show != 0]),
                stdev=np.std(image_to_show[image_to_show != 0]),
            )
        )
    return results


def compare_reference_and_spatial(
    reference_anndata,
    spatial_anndata,
    category="MTG_subclass_name",
    layer_field=None,
    plot_stuff=True,
    target_obs_key="comparison_transcript_counts",
    ok_to_clobber=False,
):
    """
    Compare reference and spatial data based on a specified category.
    Parameters:
    - reference_anndata (AnnData): Annotated data matrix containing the reference data.
    - spatial_anndata (AnnData): Annotated data matrix containing the spatial data.
    - category (str): The category to compare the data on. Default is "MTG_subclass_name".
    - layer_field (str): The field to use for layer-based comparison. Default is None.
    - plot_stuff (bool): Whether to plot the comparison results. Default is True.
    - target_obs_key (str): The key to use for storing the comparison transcript counts. Default is "comparison_transcript_counts".
    - ok_to_clobber (bool): Whether it is okay to overwrite the target_obs_key in the reference_anndata. Default is False.
    Raises:
    - ValueError: If the target_obs_key already exists in the reference_anndata and ok_to_clobber is False.
    Returns:
    - None
    """
    if target_obs_key in reference_anndata.obs.columns:
        if not ok_to_clobber == True:
            raise ValueError(
                "obs key "
                + target_obs_key
                + " is already in the input reference .obs\n If desired, set ok_to_clobber to True"
            )
        else:
            print(
                "warning: modifying input reference anndata .obs field "
                + target_obs_key
            )

    if layer_field is not None:
        reference_anndata.obs[target_obs_key] = np.sum(
            reference_anndata.layers[layer_field], axis=1
        )
    else:
        reference_anndata.obs[target_obs_key] = np.sum(reference_anndata.X, axis=1)

    means = (
        reference_anndata.obs.loc[:, [category, target_obs_key]]
        .groupby(category, observed=True)
        .mean()
    )
    means["spatial_counts"] = 0.0

    for name in spatial_anndata.obs[category].unique():
        means.loc[name, ["spatial_counts"]] = (
            spatial_anndata.obs.loc[
                spatial_anndata.obs[category] == name, [target_obs_key]
            ]
            .mean()
            .values[0]
        )

    fit_values = np.polyfit(means.spatial_counts, means[target_obs_key], 1)

    scale_factor = 1 / fit_values[0]
    if plot_stuff:
        plt.figure()
        plt.plot(means.spatial_counts, means[target_obs_key], ".")
        plt.xlabel("spatial counts")
        plt.ylabel(target_obs_key + " in reference anndata")
        plt.figure()
        sns.regplot(x=means.spatial_counts, y=scale_factor * means[target_obs_key])
        plt.plot([0, 1000], [0, 1000], label="unity")
        # plt.plot(means.spatial_counts, means.spatial_counts*fit_values[0]+fit_values[1], label = "fit")
        plt.ylabel("scaled reference data")
        plt.title("\n scale = " + str(scale_factor)[:6])
        plt.legend()

    # scale transcript counts based on the linear fit.
    # could also do this per group.

    reference_anndata.obs[target_obs_key] = (
        scale_factor * reference_anndata.obs[target_obs_key]
    )

# only works if the user has consistent annotations between the two anndatas
def returnARI(sc, seg_comp_df, seg_name_a, seg_name_b, annotation_key='annotations'):
    seg_comp_df['annotations'] = None  # Initialize the column with None or NaN

    # extract the annotations from the anndatas
    annotations_a = sc.ad_0.obs[annotation_key].tolist()
    annotations_b = sc.ad_1.obs[annotation_key].tolist()

    # Assign annotations based on the source
    seg_comp_df.loc[seg_comp_df['source'] == seg_name_a, 'annotations'] = annotations_a
    seg_comp_df.loc[seg_comp_df['source'] == seg_name_b, 'annotations'] = annotations_b

    # find all the mutually matched cells
    seg_b_df = seg_comp_df[seg_comp_df['source']==seg_name_b]
    high_q_b_cells = seg_b_df[seg_b_df['low_quality_cells']==False]
    matched_indices = high_q_b_cells.iloc[:,4].dropna().copy()

    # create the label lists of the matched cells
    labels_a = []
    labels_b = []
    for index, value in matched_indices.items():
        labels_a.append(str(seg_comp_df.loc[index]['annotations']))
        labels_b.append(str(seg_comp_df.loc[value]['annotations']))

    # calculate the ARI and return it
    ari = adjusted_rand_score(labels_a, labels_b)
    print(f"Adjusted Rand Index: {ari}")
    return ari

def get_normalized_Hausdorff(seg_comp_df, seg_name_a, seg_name_b):
    coords_a = seg_comp_df[seg_comp_df['source']==seg_name_a]
    coords_b = seg_comp_df[seg_comp_df['source']==seg_name_b]

    plt.scatter(coords_a['center_x'], coords_a['center_y'], color='blue')
    plt.scatter(coords_b['center_x'], coords_b['center_y'], color='red')
    plt.show()

    x1 = coords_a['center_x'].to_numpy()
    y1 = coords_a['center_y'].to_numpy()
    x2 = coords_b['center_x'].to_numpy()
    y2 = coords_b['center_y'].to_numpy()

    # Calculate range considering the axis
    x_coords = np.concatenate((x1, x2))
    y_coords = np.concatenate((y1, y2))
    x_data_range = np.max(x_coords) - np.min(x_coords)
    y_data_range = np.max(y_coords) - np.min(y_coords)

    average_data_range = (x_data_range+y_data_range)/2
    print(f"Average data range: {average_data_range}")

    # Calculate Hausdorff distance
    points1 = np.column_stack((x1, y1))
    points2 = np.column_stack((x2, y2))
    hausdorff_dist = distance.directed_hausdorff(points1, points2)[0]
    print(f"Hausdorff Distance: {hausdorff_dist}")
    print(f"Hausdorff Distance normalized to range (%): {(hausdorff_dist/average_data_range)*100}")
    return ((hausdorff_dist/average_data_range)*100)