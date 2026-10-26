import numpy as np
import pytest
from experiments.interoception.prefix_audit import audit_prefixes


def test_causal_threshold_decisions_keep_exact_history():
    x=np.tile([0.,1.,0.,0.],100)
    detector=lambda data,fs:np.flatnonzero((data[1:]>0)&(data[:-1]<=0))+1
    result=audit_prefixes(x,10,detector,[10,20,40],guard_s=1,start_s=1)
    assert result['informative_comparisons']==2
    assert result['comparisons_with_revised_history']==0
    assert all(row['changed_decision_memberships']==0 for row in result['comparisons'])


def test_global_future_statistic_revises_history_away_from_boundary():
    x=np.r_[np.tile([0.,1.],100),np.full(100,100.)]
    result=audit_prefixes(x,10,lambda data,fs:np.flatnonzero(data>data.mean()),[10,20,30],guard_s=1,start_s=1)
    assert result['comparisons_with_revised_history']==2
    first=result['comparisons'][0]
    assert first['historical_decisions_removed_after_append']==40
    assert first['historical_decisions_added_after_append']==0
    assert not result['independent_physiological_reference_used']


def test_terminal_changes_are_excluded_and_last_prefix_is_identity():
    result=audit_prefixes(np.ones(300),10,lambda data,fs:np.array([len(data)-1]),[10,20,30],guard_s=1)
    assert all(row['changed_decision_memberships']==0 for row in result['comparisons'])
    assert result['comparisons'][-1]['future_samples_appended']==0


@pytest.mark.parametrize('bad',[[3,3],[-1],[500],[2.5]])
def test_invalid_detector_indices_are_not_scored(bad):
    with pytest.raises(ValueError,match='decision indices'):
        audit_prefixes(np.ones(300),10,lambda data,fs:np.array(bad),[12,20])


def test_prefixes_and_values_are_checked():
    for prefixes in ([20,10],[40],[5]):
        with pytest.raises(ValueError):
            audit_prefixes(np.ones(300),10,lambda data,fs:np.array([],int),prefixes)
