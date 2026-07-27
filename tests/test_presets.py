from rrg.presets import ASSET_NAMES, PRESET_UNIVERSES, asset_name


def test_core_presets_have_unique_assets_and_valid_benchmarks():
    assert set(PRESET_UNIVERSES) == {
        "US sectors",
        "US factors",
        "Global regions",
        "Cross-asset",
    }
    for preset in PRESET_UNIVERSES.values():
        assert 1 <= len(preset.assets) <= 12
        assert len(set(preset.assets)) == len(preset.assets)
        assert preset.benchmark not in preset.assets


def test_asset_names_fall_back_to_ticker():
    assert ASSET_NAMES["XLK"] == "Technology"
    assert asset_name("xlk") == "Technology"
    assert asset_name("custom") == "CUSTOM"
