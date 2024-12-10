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


def test_get_column_ordering():
    ordered_columns = get_column_ordering(TEST_DF, ordered_rows=["a", "b", "c"])
    print(ordered_columns)
    assert ordered_columns == ["a", "c", "b"]


def test_SpatialCompare():
    # mock up test
    sc = SpatialCompare(TEST_ANNDATAS[0], TEST_ANNDATAS[1])
    assert all(sc.ad_0[0].obs.columns == sc.ad_1[1].obs.columns)
