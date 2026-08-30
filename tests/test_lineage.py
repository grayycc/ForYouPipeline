from agent.lineage import (
    FeatureProvenance,
    check_feature_lineage,
    is_random_exposure_split,
    static_metadata_feature,
    strict_past_feature,
)


def test_static_metadata_is_safe_for_all_splits():
    feature = static_metadata_feature('video_category', 'metadata')
    ok, reason = check_feature_lineage(feature, split_name='valid')
    assert ok, reason
    assert not is_random_exposure_split('valid')


def test_train_only_counts_are_safe():
    feature = strict_past_feature(
        'user_topic_count',
        split_scope='train',
        uses_label=False,
        row_scope='train_only',
    )
    ok, reason = check_feature_lineage(feature, split_name='valid')
    assert ok, reason


def test_eval_rows_cannot_be_used_to_fit():
    feature = FeatureProvenance(
        name='target_encoding',
        source='behavior',
        split_scope='valid',
        uses_label=True,
        row_scope='evaluation',
    )
    ok, reason = check_feature_lineage(feature, split_name='valid')
    assert not ok and 'future' in reason.lower(), reason


def test_random_exposure_rows_are_isolated():
    assert is_random_exposure_split('random')
    assert is_random_exposure_split('random_exposure')
