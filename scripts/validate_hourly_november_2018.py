"""Exact UTC COSMO-REA6 versus SMARD hourly validation for November 2018."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app_assets/2018"
COSMO = ASSETS / "cosmo_rea6_de_201811_hourly.parquet"
SMARD = ASSETS / "smard_de_201811_hourly.parquet"
OUT = ASSETS / "cosmo_smard_de_201811_hourly_validation.parquet"
QC_OUT = ASSETS / "cosmo_smard_de_201811_hourly_validation_qc.json"
EXPECTED = pd.date_range("2018-11-01", "2018-11-30 23:00", freq="h")
TECHS = ("wind", "solar", "hybrid")
WEIGHTINGS = ("area", "capacity")


def metrics(obs: pd.Series, model: pd.Series) -> dict:
    valid = obs.notna() & model.notna(); o, m = obs[valid], model[valid]; e = m-o
    correlation = o.corr(m) if o.nunique() > 1 and m.nunique() > 1 else None
    return {"n": int(valid.sum()), "observed_mean": float(o.mean()), "model_mean": float(m.mean()),
            "correlation": float(correlation) if pd.notna(correlation) else None,
            "bias_model_minus_observed": float(e.mean()), "mae": float(e.abs().mean()),
            "rmse": float(np.sqrt(np.mean(e**2)))}


def ramp_metrics(obs: pd.Series, model: pd.Series, time: pd.Series) -> dict:
    consecutive = time.diff().eq(pd.Timedelta(hours=1)); o, m = obs.diff()[consecutive], model.diff()[consecutive]
    out = metrics(o, m)
    return {"n_ramps": out.pop("n"), "observed_mean_ramp": out.pop("observed_mean"),
            "model_mean_ramp": out.pop("model_mean"), "ramp_correlation": out.pop("correlation"),
            "ramp_bias_model_minus_observed": out.pop("bias_model_minus_observed"),
            "ramp_mae": out.pop("mae"), "ramp_rmse": out.pop("rmse")}


def runs(flag: pd.Series, time: pd.Series) -> list[dict]:
    groups = flag.ne(flag.shift()).cumsum(); result=[]
    for _, idx in flag[flag].groupby(groups[flag]).groups.items():
        pos=list(idx); result.append({"start_utc": str(time.iloc[pos[0]]), "end_utc": str(time.iloc[pos[-1]]), "duration_hours": len(pos)})
    return result


def classification(obs: pd.Series, model: pd.Series, time: pd.Series) -> dict:
    o=obs.lt(.10); m=model.lt(.10); tp=int((o&m).sum()); tn=int((~o&~m).sum()); fp=int((~o&m).sum()); fn=int((o&~m).sum())
    oruns, mruns = runs(o,time), runs(m,time)
    return {"threshold": "hybrid CF < 0.10", "description": "low-output resource classification, not shortage validation",
            "confusion": {"true_positive":tp,"true_negative":tn,"false_positive":fp,"false_negative":fn},
            "precision": tp/(tp+fp) if tp+fp else None, "recall":tp/(tp+fn) if tp+fn else None, "agreement":(tp+tn)/len(o),
            "observed_runs":oruns, "model_runs":mruns,
            "observed_run_summary":{"count":len(oruns),"longest_hours":max((x["duration_hours"] for x in oruns),default=0)},
            "model_run_summary":{"count":len(mruns),"longest_hours":max((x["duration_hours"] for x in mruns),default=0)}}


def main():
    c=pd.read_parquet(COSMO); s=pd.read_parquet(SMARD)
    c["interval_start_utc"]=pd.to_datetime(c["time"])-pd.Timedelta(hours=1)
    s["interval_start_utc"]=pd.to_datetime(s["interval_start_utc"])
    for label, frame in (("COSMO",c),("SMARD",s)):
        idx=pd.DatetimeIndex(frame.interval_start_utc)
        if len(frame)!=720 or idx.duplicated().any() or not idx.sort_values().equals(EXPECTED): raise ValueError(f"{label} does not exactly cover 720 UTC interval starts")
    keep=["interval_start_utc","time","wind_cf_area","solar_cf_area","hybrid_cf_area","wind_cf_capacity","solar_cf_capacity","hybrid_cf_capacity","ghi_area_wm2","ghi_capacity_solar_wm2"]
    joined=c[keep].merge(s,on="interval_start_utc",how="inner",validate="one_to_one")
    if len(joined)!=720 or joined.isna().any().any(): raise ValueError("aligned table is not complete")
    joined.to_parquet(OUT,index=False,compression="zstd")
    validation={}; ramps={}; event={}; classification_results={}
    event_mask=joined.interval_start_utc.between("2018-11-01","2018-11-09 23:00")
    for weighting in WEIGHTINGS:
        validation[weighting]={}; ramps[weighting]={}
        for tech in TECHS:
            validation[weighting][tech]=metrics(joined[f"observed_{tech}_cf"],joined[f"{tech}_cf_{weighting}"])
            ramps[weighting][tech]=ramp_metrics(joined[f"observed_{tech}_cf"],joined[f"{tech}_cf_{weighting}"],joined.interval_start_utc)
        classification_results[weighting]=classification(joined.observed_hybrid_cf,joined[f"hybrid_cf_{weighting}"],joined.interval_start_utc)
        event[weighting]={tech:metrics(joined.loc[event_mask,f"observed_{tech}_cf"],joined.loc[event_mask,f"{tech}_cf_{weighting}"]) for tech in TECHS}
    obs_day=joined.solar_generation_mwh.gt(1.0); area_day=joined.ghi_area_wm2.gt(0); cap_day=joined.ghi_capacity_solar_wm2.gt(0)
    solar_strata={"definitions":{"observed_daylight":"SMARD solar generation > 1 MWh","model_daylight":"corresponding model GHI > 0 W m-2"},
      "counts":{"observed_daylight":int(obs_day.sum()),"observed_night":int((~obs_day).sum()),"area_model_daylight":int(area_day.sum()),"capacity_model_daylight":int(cap_day.sum()),
                "tiny_nonzero_observed_night_hours":int(((~obs_day)&joined.solar_generation_mwh.gt(0)).sum())},"metrics":{}}
    for w,day in (("area",area_day),("capacity",cap_day)):
        solar_strata["metrics"][w]={"observed_definition_daylight":metrics(joined.loc[obs_day,"observed_solar_cf"],joined.loc[obs_day,f"solar_cf_{w}"]),
          "observed_definition_night":metrics(joined.loc[~obs_day,"observed_solar_cf"],joined.loc[~obs_day,f"solar_cf_{w}"]),
          "model_definition_daylight":metrics(joined.loc[day,"observed_solar_cf"],joined.loc[day,f"solar_cf_{w}"]),
          "model_definition_night":metrics(joined.loc[~day,"observed_solar_cf"],joined.loc[~day,f"solar_cf_{w}"])}
    qc={"records":len(joined),"exact_matches":len(joined),"missing_cosmo":0,"missing_smard":0,"duplicates":0,"start_interval_start_utc":str(joined.interval_start_utc.min()),"end_interval_start_utc":str(joined.interval_start_utc.max()),
        "alignment":"COSMO valid-time minus one hour equals timezone-naive UTC SMARD interval_start_utc; e.g. COSMO 2018-11-01 01:00 -> interval start 2018-11-01 00:00 UTC.",
        "validation_metrics":validation,"ramp_validation":ramps,"solar_daylight_night":solar_strata,"low_output_classification":classification_results,
        "event_window_2018_11_01_through_09":event,"data_leakage":"None: no fitting, scaling, calibration, or threshold tuning used observed data.",
        "caveats":["Observed hybrid is hypothetical equal-rated-capacity CF, not actual combined generation.","Static end-2018 capacity denominators slightly understate observed CF when capacity was lower."]}
    QC_OUT.write_text(json.dumps(qc,indent=2),encoding="utf-8"); print(f"Wrote {len(joined)} aligned hours to {OUT}")
if __name__=="__main__": main()
