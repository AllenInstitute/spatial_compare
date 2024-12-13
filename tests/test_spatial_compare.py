from spatial_compare import SpatialCompare, get_column_ordering
import pandas as pd
import anndata as ad
import pathlib

SC_DIR = pathlib.Path().absolute()
print(SC_DIR)

DATA_STEMS = [
    "CJ_BG_mini1.h5ad",
    "CJ_BG_mini2.h5ad",
    "CJ_BG_mini3.h5ad",
    "CJ_BG_mini4.h5ad",
    "sc_eg_mini_a.h5ad",
    "sc_eg_mini_a.h5ad",
]

TEST_DIR = SC_DIR.joinpath("tests").joinpath("data")

TEST_ANNDATAS = [ad.read_h5ad(TEST_DIR.joinpath(DATA_STEM)) for DATA_STEM in DATA_STEMS]

TEST_DF_RECORDS = [
    dict(a=1, b=0.5, c=0.1),
    dict(a=0.5, b=0.8, c=0.9),
    dict(a=0.1, b=1.0, c=0.1),
]
TEST_DF = pd.DataFrame.from_records(TEST_DF_RECORDS)
TEST_DF.index = ["a", "b", "c"]

TEST_SEG_NAMES = ["XEN", "SIS"]


def test_get_column_ordering():
    ordered_columns = get_column_ordering(TEST_DF, ordered_rows=["a", "b", "c"])
    print(ordered_columns)
    assert ordered_columns == ["a", "c", "b"]


def test_SpatialCompare():
    # mock up test
    sc = SpatialCompare(TEST_ANNDATAS[0], TEST_ANNDATAS[1])
    assert all(sc.ad_0[0].obs.columns == sc.ad_1[1].obs.columns)


def test_segmentation_comparison():
    sc = SpatialCompare(
        TEST_ANNDATAS[4],
        TEST_ANNDATAS[5],
        data_names=[TEST_SEG_NAMES[0], TEST_SEG_NAMES[1]],
        obsm_key="spatial",
    )
    seg_comp_df = sc.collect_mutual_match_and_doublets(
        bc="1370519421", save=False, reuse_saved=False, savepath=TEST_DIR
    )
    seg_a_df = seg_comp_df[seg_comp_df["source"] == TEST_SEG_NAMES[0]]
    seg_b_df = seg_comp_df[seg_comp_df["source"] == TEST_SEG_NAMES[1]]
    assert len(TEST_ANNDATAS[4]) == len(seg_a_df)
    assert len(TEST_ANNDATAS[5]) == len(seg_b_df)

    high_q_a_cells = seg_a_df[seg_a_df["low_quality_cells"] == False]
    high_q_b_cells = seg_b_df[seg_b_df["low_quality_cells"] == False]
    assert len(high_q_a_cells.iloc[:, 4].dropna()) == len(
        high_q_b_cells.iloc[:, 4].dropna()
    )

    matched_cells_a = (
        seg_comp_df[seg_comp_df["source"] == TEST_SEG_NAMES[0]].iloc[:, 4].dropna()
    )
    matched_cells_b = (
        seg_comp_df[seg_comp_df["source"] == TEST_SEG_NAMES[1]].iloc[:, 4].dropna()
    )
    assert (
        seg_comp_df.loc[
            matched_cells_a.index.values.tolist(), seg_comp_df.columns[4]
        ].values.tolist()
        == matched_cells_a.values.tolist()
    )
