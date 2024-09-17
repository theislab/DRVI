from drvi.nn_modules.feature_interface import FeatureInfo, FeatureInfoList


class TestFeatureInfoList:
    def test_fill_with_total_dim(self):
        fil = FeatureInfoList(["x@16!tada", "y", "z!zizi!gulu", "w@8"], total_dim=32)
        assert fil.feature_info_list[0] == FeatureInfo("x", 16, ("tada",))
        assert fil.feature_info_list[1] == FeatureInfo("y", 4, ())
        assert fil.feature_info_list[2] == FeatureInfo("z", 4, ("zizi", "gulu"))
        assert fil.feature_info_list[3] == FeatureInfo("w", 8, ())

        assert tuple(fil.names) == ("x", "y", "z", "w")
        assert tuple(fil.dims) == (16, 4, 4, 8)
        assert tuple(fil.keywords_list) == (("tada",), (), ("zizi", "gulu"), ())
