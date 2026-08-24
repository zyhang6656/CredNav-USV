from scripts import check_mixed_cache_hits


def test_resolve_datasets_uses_configured_training_sources_and_eval_bundle():
    config = {
        "data": {
            "train_sources": [
                {
                    "label": "obs3",
                    "scenario_dir": "simple_boat/assets/nav3_new_map",
                    "kf_cache_dir": "results/kf_cache/obs3",
                },
                {
                    "label": "obs4",
                    "scenario_dir": "simple_boat/assets/nav4_new_map",
                    "kf_cache_dir": "results/kf_cache/obs4",
                },
            ],
            "eval_scenario_dir": "simple_boat/assets/mixed_eval3_6_new_map_20260707",
        },
        "cache": {"kf_cache_dir": ["results/kf_cache/obs3", "results/kf_cache/obs4"]},
    }

    assert check_mixed_cache_hits.resolve_datasets(config) == [
        ("train_obs3", "simple_boat/assets/nav3_new_map", "results/kf_cache/obs3"),
        ("train_obs4", "simple_boat/assets/nav4_new_map", "results/kf_cache/obs4"),
        (
            "eval",
            "simple_boat/assets/mixed_eval3_6_new_map_20260707",
            ["results/kf_cache/obs3", "results/kf_cache/obs4"],
        ),
    ]
