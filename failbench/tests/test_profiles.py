"""Tests for the Nav2 MPPI tuning profiles and the record of changes."""

from dataclasses import replace

import pytest
from failbench.mppi.nav2 import Nav2MppiCfg
from failbench.mppi.profiles import Change, apply_changes, changes_from_stock


def test_stock_settings_record_no_changes():
  assert changes_from_stock(Nav2MppiCfg()) == []


def test_every_changed_parameter_is_recorded_with_its_stock_value():
  stock = Nav2MppiCfg()
  critics = replace(
    stock.critics, path_align=replace(stock.critics.path_align, weight=7.0)
  )
  cfg = replace(stock, vy_std=0.4, critics=critics)
  assert changes_from_stock(cfg) == [
    {"param": "vy_std", "stock": 0.2, "value": 0.4},
    {"param": "critics.path_align.weight", "stock": 14.0, "value": 7.0},
  ]


def test_changes_apply_by_dotted_name_and_reject_unknown_ones():
  cfg = apply_changes(
    Nav2MppiCfg(),
    (
      Change("vy_std", 0.4, "wider sideways samples"),
      Change("critics.path_align.weight", 7.0, "let plans leave the line"),
    ),
  )
  assert cfg.vy_std == 0.4 and cfg.critics.path_align.weight == 7.0
  assert cfg.critics.path_follow.weight == Nav2MppiCfg().critics.path_follow.weight
  with pytest.raises(AttributeError):
    apply_changes(Nav2MppiCfg(), (Change("critics.nope.weight", 1.0, ""),))
