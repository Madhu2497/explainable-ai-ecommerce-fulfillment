"""Supplementary audit: label maturity and knowledge-calibrator bias.

Run: python label_maturity_sensitivity.py --data archive.zip --out results
Competing calibration cohorts share the SAME base XGBoost fitted before the
older calibration month; no held-out month data are used in fitting or selection.
No algorithm is selected on the held-out results. Designed for diagnostic use.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss
from xgboost import XGBClassifier
from scipy.optimize import minimize
from scipy.special import expit,logit
from experiment import load_features, BASE, SEED
from decision_validation import correction_fit, apply_correction, RULES
MONTHS=pd.date_range('2018-03-01','2018-07-01',freq='MS')

def build_xgb():
    return XGBClassifier(n_estimators=260,max_depth=3,learning_rate=.045,
        subsample=.85,colsample_bytree=.9,objective='binary:logistic',
        eval_metric='logloss',tree_method='hist',n_jobs=4,random_state=SEED)

def get_p_and_rules(model,imputer,frame):
    x=imputer.transform(frame[BASE]); p=model.predict_proba(x)[:,1]
    rule=frame[RULES].fillna(0).to_numpy(dtype=float)
    return p,rule

def run(data,out):
    records=[]
    maturity_rows=[]
    for cutoff in MONTHS:
        old_begin=cutoff-pd.DateOffset(months=3)
        old_end=old_begin+pd.DateOffset(months=1)
        recent_begin=cutoff-pd.DateOffset(months=1)
        test_end=cutoff+pd.DateOffset(months=1)
        # At the first minute of the older validation month, train only on
        # already completed orders. Never train on either calibration cohort.
        tr=data[(data.order_purchase_timestamp <old_begin)&(data.order_delivered_customer_date<old_begin)]
        old_all=data[(data.order_purchase_timestamp>=old_begin)&(data.order_purchase_timestamp<old_end)]
        recent_all=data[(data.order_purchase_timestamp>=recent_begin)&(data.order_purchase_timestamp<cutoff)]
        va_lag=old_all[old_all.order_delivered_customer_date<cutoff]
        va_recent=recent_all[recent_all.order_delivered_customer_date<cutoff]
        maturity_rows.append(dict(
            test_month=cutoff.strftime('%Y-%m'),
            recent_cohort_n=len(recent_all),
            observed_recent_n=len(va_recent),
            observed_recent_fraction=len(va_recent)/len(recent_all),
            observed_recent_late_rate=float(va_recent.late.mean()),
            eventual_recent_late_rate=float(recent_all.late.mean()),
            lagged_cohort_n=len(old_all),
            observed_lagged_n=len(va_lag),
            observed_lagged_fraction=len(va_lag)/len(old_all),
            observed_lagged_late_rate=float(va_lag.late.mean())
        ))
        te=data[(data.order_purchase_timestamp>=cutoff)&(data.order_purchase_timestamp<test_end)]
        if min(map(len,[tr,va_lag,va_recent,te]))<100 or te.late.sum()<10:
            raise RuntimeError('Insufficient observations in '+str(cutoff))
        assert tr.order_delivered_customer_date.max()<old_begin
        assert va_lag.order_purchase_timestamp.min()>=old_begin
        assert va_recent.order_purchase_timestamp.min()>=recent_begin
        assert va_lag.order_delivered_customer_date.max()<cutoff
        assert va_recent.order_delivered_customer_date.max()<cutoff
        assert te.order_purchase_timestamp.min()>=cutoff
        imp=SimpleImputer(strategy='median').fit(tr[BASE])
        mdl=build_xgb().fit(imp.transform(tr[BASE]),tr.late.to_numpy())
        yp=te.late.to_numpy()
        test_p,test_r=get_p_and_rules(mdl,imp,te)
        preds={'Raw XGBoost':test_p}
        for tag,val in [('Recent observed',va_recent),('Lagged mature',va_lag)]:
            vp,vr=get_p_and_rules(mdl,imp,val)
            vy=val.late.to_numpy()
            a=correction_fit(vp,vy,vr,with_rules=False)
            b=correction_fit(vp,vy,vr,with_rules=True)
            preds[f'{tag}: global calibration']=apply_correction(test_p,test_r,a)
            preds[f'{tag}: knowledge overlay']=apply_correction(test_p,test_r,b)
        for name,p in preds.items():
            records.append(dict(test_month=cutoff.strftime('%Y-%m'),method=name,
               train_n=len(tr),test_n=len(te),test_late_n=int(yp.sum()),
               test_late_rate=float(yp.mean()),
               recent_known_n=len(va_recent),recent_total_n=len(recent_all),
               recent_known_fraction=float(len(va_recent)/len(recent_all)),
               recent_observed_late_rate=float(va_recent.late.mean()),
               lagged_known_n=len(va_lag),lagged_total_n=len(old_all),
               lagged_known_fraction=float(len(va_lag)/len(old_all)),
               lagged_observed_late_rate=float(va_lag.late.mean()),
               Brier=brier_score_loss(yp,p),AP=average_precision_score(yp,p),
               delta_Brier_vs_raw=brier_score_loss(yp,p)-brier_score_loss(yp,test_p)))
        print(cutoff.strftime('%Y-%m'),'recent known %.1f%%'%(len(va_recent)*100/len(recent_all)),
          'mature known %.1f%%'%(len(va_lag)*100/len(old_all)),
          'Brier',', '.join(f'{name}: {brier_score_loss(yp,p):.5f}' for name,p in preds.items()),flush=True)
    results=pd.DataFrame(records)
    out.mkdir(parents=True,exist_ok=True)
    results.to_csv(out/'maturity_sensitivity.csv',index=False,float_format='%.8f')
    pd.DataFrame(maturity_rows).to_csv(out/'outcome_maturity_audit.csv',index=False,float_format='%.8f')
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('results'));args=p.parse_args()
    run(load_features(args.data),args.out)