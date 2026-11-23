"""Audit regression against the actual recorded same-exposure schema."""
import csv
from pathlib import Path
import statistics
import pytest
from stage16_evidence_audit import same_exposure_stats


def test_all_recorded_doses_include_measured_centroid_shift():
    path=Path(__file__).parent/'stage15_evidence/noise/same_exposure_dose.csv'
    with path.open() as f:
        rows=list(csv.DictReader(f))
    result=same_exposure_stats(rows)
    assert len(result)==5
    for record in result:
        measured=[float(r['centroid_shift_px']) for r in rows if float(r['sigma_dn'])==record['sigma_dn'] and r['centroid_shift_px']]
        assert record['rows']==72
        assert record['centroid_shift_px_median']==statistics.median(measured)
        assert record['centroid_shift_px_measured_rows']==len(measured)
        assert record['centroid_shift_px_missing_rows']==72-len(measured)
    assert result[0]['centroid_shift_px_median']==0
    assert result[-1]['centroid_shift_px_median']>0


def test_missing_center_metric_is_an_explicit_schema_error():
    with pytest.raises(KeyError, match='centroid_shift_px'):
        same_exposure_stats([dict(sigma_dn=3,mask_iou=.9,mask_boundary_iou=.8)])


def test_invisible_centers_remain_missing():
    rows=[dict(sigma_dn=3,mask_iou=.9,mask_boundary_iou=.8,centroid_shift_px='')]
    result=same_exposure_stats(rows)[1]
    assert result['centroid_shift_px_median'] is None
    assert result['centroid_shift_px_measured_rows']==0
    assert result['centroid_shift_px_missing_rows']==1
