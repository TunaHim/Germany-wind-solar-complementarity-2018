"""Generate and execute notebook 06: hourly resolution and observed SMARD validation."""
from pathlib import Path
from textwrap import dedent
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
ROOT=Path(__file__).resolve().parent.parent; ASSETS=ROOT/"app_assets/2018"; OUTPUT=ROOT/"notebooks/06_germany_2018_hourly_case_study.ipynb"
def md(x): return nbformat.v4.new_markdown_cell(dedent(x).strip())
def code(x): return nbformat.v4.new_code_cell(dedent(x).strip())
def main():
    required=[ASSETS/x for x in ["cosmo_rea6_de_201811_hourly.parquet","cosmo_rea6_de_201811_hourly_qc.json","smard_de_201811_hourly.parquet","smard_de_201811_hourly_qc.json","cosmo_smard_de_201811_hourly_validation.parquet","cosmo_smard_de_201811_hourly_validation_qc.json"]]
    if any(not x.exists() for x in required): raise SystemExit("Missing hourly artifacts; run SMARD download and validation scripts first")
    cells=[md("""# 06 — November 2018 hourly resolution case study and observed validation

This is a **one-month hourly case study**, not a full-year hourly analysis. It reads only committed compact artifacts and compares COSMO-REA6 area-weighted resource and capacity-weighted modelled fleet proxies with SMARD observed national grid-fed generation.

**Result to keep in view:** solar hourly timing and ramps are reproduced strongly, whereas wind ramp correlation is much weaker. Capacity weighting improves some wind and hybrid metrics but does not solve the wind-ramp problem. The preserved hourly-versus-daily comparison also demonstrates that nonlinear wind conversion and temporal averaging do not commute.

SMARD is the primary national observational reference; OPSD supplies spatial installed-capacity weights, not generation observations. OPSD time series are not treated as independent validation because provenance may overlap operational sources."""),
    md("""## Provenance, units, denominator, and clock

SMARD filters 4067 (onshore wind) and 4068 (PV) provide MWh in each one-hour interval; attribution: **Bundesnetzagentur | SMARD.de, CC BY 4.0**, underlying data received from ENTSO-E. CF divides MWh by fixed end-2018 capacity (52,565 MW wind; 45,277 MW solar) times 1 h. This static denominator slightly understates CF when capacity was lower; correlation and ramp timing are less affected than level. Observed hybrid CF is `0.5 wind + 0.5 solar`: a hypothetical equal-rated-capacity portfolio, not actual combined national generation.

COSMO labels hours ending from 2018-11-01 01:00 through 2018-12-01 00:00 UTC. We subtract one hour and match SMARD **in UTC**, never by Berlin wall time. Numerically, COSMO `2018-11-01 01:00` → interval start `2018-11-01 00:00 UTC` → Berlin display `2018-11-01T01:00:00+01:00`. November has no DST transition, but clocks remain distinct."""),
    code("""%matplotlib inline
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path.cwd(); A=ROOT/'app_assets'/'2018'
aligned=pd.read_parquet(A/'cosmo_smard_de_201811_hourly_validation.parquet')
with open(A/'cosmo_smard_de_201811_hourly_validation_qc.json',encoding='utf-8') as f: vqc=json.load(f)
with open(A/'cosmo_rea6_de_201811_hourly_qc.json',encoding='utf-8') as f: cqc=json.load(f)
aligned['interval_start_utc']=pd.to_datetime(aligned.interval_start_utc)
pd.Series({'COSMO rows':720,'SMARD rows':vqc['records'],'exact UTC matches':vqc['exact_matches'],'missing COSMO':vqc['missing_cosmo'],'missing SMARD':vqc['missing_smard'],'duplicates':vqc['duplicates'],'first':vqc['start_interval_start_utc'],'last':vqc['end_interval_start_utc']},name='QC')"""),
    md("""## Hourly versus daily processing routes

Nonlinear weather-to-CF conversion applied hourly and then averaged need not equal conversion of daily-mean weather. The COSMO QC summary below compares those two processing routes; it is resolution sensitivity, not observation skill."""),
    code("pd.DataFrame(cqc['daily_vs_existing_daily']['metrics']).T.round(4)"),
    md("""## Observed and modelled hourly trajectories

Area weighting describes Germany's land resource; OPSD capacity weighting emphasizes mapped 2018 fleet locations. Both remain generic model proxies."""),
    code("""fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True)
for ax,t in zip(axes,['wind','solar']):
 ax.plot(aligned.interval_start_utc,aligned[f'observed_{t}_cf'],label='SMARD observed',c='black',lw=1.5)
 ax.plot(aligned.interval_start_utc,aligned[f'{t}_cf_area'],label='COSMO area',alpha=.75)
 ax.plot(aligned.interval_start_utc,aligned[f'{t}_cf_capacity'],label='COSMO capacity',alpha=.75)
 ax.set(title=t.title(),ylabel='CF'); ax.grid(alpha=.25); ax.legend(ncol=3)
plt.tight_layout(); plt.show()"""),
    code("""fig,axes=plt.subplots(2,2,figsize=(10,9))
for i,t in enumerate(['wind','solar']):
 for j,w in enumerate(['area','capacity']):
  ax=axes[i,j]; ax.scatter(aligned[f'observed_{t}_cf'],aligned[f'{t}_cf_{w}'],s=8,alpha=.45)
  lim=max(ax.get_xlim()[1],ax.get_ylim()[1]); ax.plot([0,lim],[0,lim],'k--'); ax.set(xlabel='observed CF',ylabel='model CF',title=f'{t} — {w}'); ax.grid(alpha=.2)
plt.tight_layout(); plt.show()"""),
    md("""## Level and timing metrics

Correlation primarily describes co-timing and shape; bias/MAE/RMSE describe amplitude error. High correlation does not imply an unbiased level. No observed values were used for fitting, rescaling, calibration, or threshold tuning."""),
    code("""rows=[]
for w,d in vqc['validation_metrics'].items():
 for t,m in d.items(): rows.append({'weighting':w,'technology':t,**m})
pd.DataFrame(rows).set_index(['weighting','technology']).round(4)"""),
    md("""## Ramp validation

Ramps are first differences over consecutive UTC hours only (719 pairs); no difference is taken across a gap. Ramp correlation evaluates transition timing/shape, while ramp bias and errors evaluate change amplitude. The results below intentionally foreground the contrast: solar ramps track observations closely, while wind ramp correlation is weak under both weighting methods."""),
    code("""rows=[]
for w,d in vqc['ramp_validation'].items():
 for t,m in d.items(): rows.append({'weighting':w,'technology':t,**m})
pd.DataFrame(rows).set_index(['weighting','technology']).round(4)"""),
    code("""fig,axes=plt.subplots(2,1,figsize=(13,6),sharex=True)
for ax,t in zip(axes,['wind','solar']):
 ax.plot(aligned.interval_start_utc,aligned[f'observed_{t}_cf'].diff(),label='observed',c='black')
 ax.plot(aligned.interval_start_utc,aligned[f'{t}_cf_capacity'].diff(),label='capacity proxy',alpha=.7)
 ax.set(ylabel='Δ CF h⁻¹',title=f'{t.title()} ramps'); ax.grid(alpha=.2); ax.legend()
plt.tight_layout(); plt.show()"""),
    md("""## Solar daylight and night

Two transparent definitions are reported rather than silently choosing one: observed daylight is SMARD PV generation >1 MWh; model daylight is GHI >0 W m⁻². SMARD can contain tiny nonzero nighttime PV values, plausibly reflecting operational reporting/aggregation effects; they should not be interpreted as solar irradiance. For the model-defined night subset the model solar CF is identically zero, so a correlation is mathematically undefined and is shown as "undefined"."""),
    code("""display(pd.Series(vqc['solar_daylight_night']['counts']))
daylight = pd.concat({w: pd.DataFrame(x).T for w, x in vqc['solar_daylight_night']['metrics'].items()}).round(5)
daylight.astype(object).where(daylight.notna(), "undefined (model = 0)")"""),
    md("""## Hybrid low-output classification and contiguous runs

The observed and model hypothetical equal-rated-capacity hybrid threshold is CF < 0.10. Confusion counts, precision, recall, agreement, and run summaries compare low-output screening classifications. Nighttime solar zeros naturally influence contiguous hourly runs, so these are not equivalent to the ≥2-day daily event catalogue. This is **not electricity-shortage or adequacy validation**: demand, storage, imports, transmission, dispatchable generation, curtailment, and outages are absent."""),
    code("""rows=[]
for w,x in vqc['low_output_classification'].items(): rows.append({'weighting':w,**x['confusion'],'precision':x['precision'],'recall':x['recall'],'agreement':x['agreement'],'observed runs':x['observed_run_summary']['count'],'observed longest h':x['observed_run_summary']['longest_hours'],'model runs':x['model_run_summary']['count'],'model longest h':x['model_run_summary']['longest_hours']})
pd.DataFrame(rows).set_index('weighting').round(3)"""),
    md("""## Event window: 1–9 November

The window is fixed a priori from the existing case study. Metrics are artifact-derived and no bias fitting is performed."""),
    code("""win=aligned[aligned.interval_start_utc.between('2018-11-01','2018-11-09 23:00')]
fig,ax=plt.subplots(figsize=(14,5)); ax.plot(win.interval_start_utc,win.observed_hybrid_cf,label='observed equal-capacity hybrid',c='black',lw=2); ax.plot(win.interval_start_utc,win.hybrid_cf_area,label='area'); ax.plot(win.interval_start_utc,win.hybrid_cf_capacity,label='capacity'); ax.axhline(.10,c='red',ls='--'); ax.set(ylabel='hybrid CF',title='Fixed event window (UTC)'); ax.grid(alpha=.25); ax.legend(); plt.show()
pd.concat({w:pd.DataFrame(x).T for w,x in vqc['event_window_2018_11_01_through_09'].items()}).round(4)"""),
    md("""## Limitations and conclusion

This is one nationally aggregated month with no plant-level observations. The conversion uses generic turbine/PV assumptions; COSMO is a reanalysis proxy, and SMARD/ENTSO-E generation can reflect curtailment, availability, reporting, and other operational effects. End-year capacity denominators are static. OPSD plant locations are spatial weights—not independent generation validation—and its time-series package is not used as a second independent reference because provenance may overlap. There is no independent OPSD generation validation.

Hourly observations support timing, ramp, daylight, and low-output classification checks, while the daily-route comparison shows what temporal resolution changes. The hypothetical equal-capacity hybrid supports resource screening only, not adequacy or shortage claims.""")]
    nb=nbformat.v4.new_notebook(cells=cells); nb.metadata['kernelspec']={'display_name':'renewable-diagnostics','language':'python','name':'python3'}
    ExecutePreprocessor(timeout=1200,kernel_name='python3').preprocess(nb,{'metadata':{'path':str(ROOT)}}); OUTPUT.parent.mkdir(exist_ok=True); nbformat.write(nb,OUTPUT); print(f'Executed notebook -> {OUTPUT}')
if __name__=='__main__': main()
