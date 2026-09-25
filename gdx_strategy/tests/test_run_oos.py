"""OOS runner: hash gate, IS-file check, seam checks, warm-up and no look-ahead.

The OOS workbook here is a synthetic, seeded continuation of the IS prices written
in the same format (temp dir only) -- it exercises the plumbing, not performance.
"""
import json
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import run_oos  # noqa: E402

from src.data import SeamError, load_dataset, load_is_oos, load_prices  # noqa: E402
from src.factor_model import ResidualCache  # noqa: E402
from src.forecast import ForecastCache  # noqa: E402

FROZEN = ROOT / "results" / "frozen_config.json"


def write_workbook(path, prices: pd.DataFrame, fedfunds: pd.DataFrame):
    with pd.ExcelWriter(path) as w:
        prices.reset_index().to_excel(w, sheet_name="Prices (input)", startrow=1, index=False)
        fedfunds.to_excel(w, sheet_name="FedFunds (input)", startrow=1, index=False)


@pytest.fixture(scope="module")
def oos_files(is_workbook, tmp_path_factory):
    d = tmp_path_factory.mktemp("oos")
    px = load_prices(is_workbook)
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", "2022-12-30", name="Date")
    rets = rng.normal(0, 0.012, (len(dates), px.shape[1]))
    new = pd.DataFrame(px.iloc[-1].to_numpy() * np.exp(np.cumsum(rets, 0)), index=dates, columns=px.columns)
    new["GDXJ"] = 1.0
    ff = pd.read_excel(is_workbook, sheet_name="FedFunds (input)", header=1)
    ok = d / "oos_ok.xlsx"
    write_workbook(ok, new, ff)
    overlap = d / "oos_overlap.xlsx"
    write_workbook(overlap, pd.concat([px.iloc[-3:].assign(GDXJ=1.0), new]), ff)
    bad_overlap = d / "oos_bad_overlap.xlsx"
    tweaked = px.iloc[-3:].assign(GDXJ=1.0)
    tweaked.iloc[1, tweaked.columns.get_loc("GDX")] *= 1.01
    write_workbook(bad_overlap, pd.concat([tweaked, new]), ff)
    gap = d / "oos_gap.xlsx"
    write_workbook(gap, new.loc["2022-01-20":], ff)
    return dict(ok=ok, overlap=overlap, bad_overlap=bad_overlap, gap=gap, n_oos=len(dates), dir=d)


def test_seam_checks(is_workbook, oos_files):
    _, r_ok, s_ok = load_is_oos(is_workbook, oos_files["ok"])
    _, r_ov, s_ov = load_is_oos(is_workbook, oos_files["overlap"])
    assert s_ok["n_oos_rows"] == s_ov["n_oos_rows"] == oos_files["n_oos"]
    assert s_ov["overlap_dropped"] == 3 and s_ok["weekdays_between"] == 0  # 2021-12-31 Fri -> 2022-01-03 Mon
    pd.testing.assert_frame_equal(r_ok, r_ov)
    with pytest.raises(SeamError):
        load_is_oos(is_workbook, oos_files["bad_overlap"])
    with pytest.raises(SeamError):
        load_is_oos(is_workbook, oos_files["gap"])
    # IS rows are exactly the IS-only returns (prior-month RF, same FEDFUNDS)
    _, r_is = load_dataset(is_workbook, check_adjusted=False)
    pd.testing.assert_frame_equal(r_ok.loc[r_is.index], r_is)


def test_oos_forecasts_are_warm_and_do_not_touch_is(is_workbook, oos_files):
    frozen = json.loads(FROZEN.read_text())
    fx = frozen["fixed"]
    p = frozen["primary"]["params"]
    _, r_all, seam = load_is_oos(is_workbook, oos_files["ok"])
    _, r_is = load_dataset(is_workbook, check_adjusted=False)

    def fc(r):
        c = ForecastCache(ResidualCache(r, min_obs_frac=fx["stage1_min_obs_frac"]),
                          window=fx["stage2_window"], min_obs=fx["stage2_min_obs"])
        return c.forecast(tuple(p["factors"]), p["lookback"], p["m"])

    a, b = fc(r_all), fc(r_is)
    pd.testing.assert_frame_equal(a.loc[r_is.index], b)              # IS unaffected by OOS data
    oos = a.loc[seam["oos_first"]:]
    assert oos[["z", "gamma", "r_hat"]].notna().all().all()           # live from OOS day 1
    assert (oos["n_pairs"].diff().dropna() == 1).all()                # gamma keeps updating


def test_run_oos_end_to_end_and_refusals(is_workbook, oos_files):
    out = oos_files["dir"] / "out"
    res = run_oos.main(["--is", str(is_workbook), "--oos", str(oos_files["ok"]),
                        "--config", str(FROZEN), "--out", str(out)])
    m = res["oos_metrics"]
    frozen = json.loads(FROZEN.read_text())
    assert list(m.index) == [frozen["primary"]["config_id"]] + [a["config_id"] for a in frozen["alternates"]]
    assert (m["n_days"] == oos_files["n_oos"]).all()                  # OOS-only metrics
    for f in ("report.html", "oos_metrics.csv", "is_vs_oos.csv", "run_manifest.json", "oos_daily_primary.csv"):
        assert (out / f).exists()
    man = json.loads((out / "run_manifest.json").read_text())
    assert max(abs(v) for v in man["is_dev_sharpe_repro_error"].values()) < 1e-6
    # positions: flat before the last IS close
    g = res["grid"]
    assert (g.pos[:, np.asarray(g.dates < pd.Timestamp(man["seam"]["is_last"]))] == 0).all()

    # modified config -> refuse
    bad = oos_files["dir"] / "frozen_config.json"
    cfg = json.loads(FROZEN.read_text())
    cfg["primary"]["params"]["k"] = 1.0
    bad.write_text(json.dumps(cfg, indent=2))
    shutil.copy(FROZEN.with_suffix(".sha256"), bad.with_suffix(".sha256"))
    with pytest.raises(SystemExit):
        run_oos.main(["--is", str(is_workbook), "--oos", str(oos_files["ok"]), "--config", str(bad),
                      "--out", str(out)])
    with pytest.raises(run_oos.FrozenConfigError):
        run_oos.verify_config(bad)
    # wrong IS file -> refuse
    with pytest.raises(SystemExit):
        run_oos.main(["--is", str(oos_files["ok"]), "--oos", str(oos_files["ok"]), "--config", str(FROZEN),
                      "--out", str(out)])
