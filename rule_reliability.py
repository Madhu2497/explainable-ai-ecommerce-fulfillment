"""Exploratory temporal rule-validity gate for Olist e-commerce delays.

Prerequisites: same directory as repository experiment.py, Olist ZIP archive.
Run: python rule_reliability.py --data archive.zip --out results

Primary safeguard: each test month's outcomes are untouched until final scoring.
Within preceding-month validation, earlier cases estimate rule residuals; later
cases decide whether to enable the rule overlay. Non-positive improvements
trigger fallback to the unaltered XGBoost model. The rules are hypotheses, NOT
carrier SLAs or validated causal mechanisms. All diagnostics are supplementary.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
from experiment import load_features, BASE, SEED

RULES = ('tight_route_commitment', 'multi_seller_coordination', 'heavy_package_flag')
MONTHS = pd.date_range('2018-03-01', '2018-07-01', freq='MS')
MIN_FLAG_SUPPORT = 30
MIN_FLAG_POSITIVES = 5
L2 = 0.04
IMPROVEMENT_FLOOR = 0.0001
EPS = 1e-6


def model():
    return XGBClassifier(n_estimators=260, max_depth=3, learning_rate=.045,
                         subsample=.85, colsample_bytree=.9, objective='binary:logistic',
                         eval_metric='logloss', tree_method='hist', n_jobs=4,random_state=SEED)


def estimate_residuals(p, y, rules, selected):
    """Fit signed, L2-regularized residual corrections on earlier calibration only.
    Fix original XGBoost logit slope at 1 to avoid confounding rank-shift effects
    from global recalibration. Offset is estimated jointly, not post-treatment.
    """
    x = np.column_stack([np.ones(len(p)), rules[:, selected]])
    base = logit(np.clip(p, EPS, 1-EPS))
    yy = np.asarray(y, dtype=float)
    def objective(v):
        q = expit(base + x@v)
        loss = np.mean(np.logaddexp(0., base+x@v) - yy*(base+x@v))
        grad=x.T@(q-yy)/len(yy)
        penalty=L2*(v@v)
        return loss+penalty,grad+2*L2*v
    result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',bounds=[(-3.,3.)]*(x.shape[1]),options={'maxiter':500})
    if not result.success: raise RuntimeError('Rule residual calibration did not converge: '+str(result.message))
    return result.x


def predict(p, rules, selected, coefs):
    x=np.column_stack([np.ones(len(p)),rules[:,selected]])
    return expit(logit(np.clip(p,EPS,1-EPS)) + x@coefs)


def metrics(y,p):
    yy=np.asarray(y,dtype=int)
    idx=np.argsort(-p,kind='mergesort')[:max(1,int(np.ceil(.1*len(yy))))]
    return dict(AP=average_precision_score(yy,p), ROC_AUC=roc_auc_score(yy,p),
                Brier=brier_score_loss(yy,p), recall_at_10pct=float(yy[idx].sum()/yy.sum()))


def run(df, out):
    rows=[]; audits=[]
    for month in MONTHS:
        validation_start = month - pd.DateOffset(months=1)
        end = month + pd.DateOffset(months=1)
        train = df.loc[(df.order_purchase_timestamp<validation_start)&
                       (df.order_delivered_customer_date<validation_start)]
        val = df.loc[(df.order_purchase_timestamp>=validation_start)&
                     (df.order_purchase_timestamp<month)&
                     (df.order_delivered_customer_date<month)].sort_values('order_purchase_timestamp')
        test = df.loc[(df.order_purchase_timestamp>=month)&
                      (df.order_purchase_timestamp<end)]
        if min(len(train),len(val),len(test))<100 or test.late.sum()<10:
            raise ValueError('Insufficient sample at '+str(month))
        imputer=SimpleImputer(strategy='median')
        x_train=imputer.fit_transform(train[list(BASE)])
        x_val=imputer.transform(val[list(BASE)])
        x_test=imputer.transform(test[list(BASE)])
        classifier=model();classifier.fit(x_train, train.late.to_numpy())
        pv=classifier.predict_proba(x_val)[:,1]
        pt=classifier.predict_proba(x_test)[:,1]
        vv=val.late.to_numpy(dtype=int)
        yt=test.late.to_numpy(dtype=int)
        rv=val[list(RULES)].fillna(0).to_numpy(dtype=int)
        rt=test[list(RULES)].fillna(0).to_numpy(dtype=int)
        cut=int(.60*len(val))
        if cut<100 or len(val)-cut<100:raise ValueError('Insufficient nested validation')
        y_cal,y_gate=vv[:cut],vv[cut:]
        r_cal,r_gate=rv[:cut],rv[cut:]
        p_cal,p_gate=pv[:cut],pv[cut:]
        supported=[]
        for j, name in enumerate(RULES):
            fired=(r_cal[:,j]==1)
            n=int(fired.sum()); positives=int(y_cal[fired].sum())
            sufficient=(n>=MIN_FLAG_SUPPORT and positives>=MIN_FLAG_POSITIVES and
                        n-positives>=MIN_FLAG_POSITIVES)
            audits.append(dict(test_month=month.strftime('%Y-%m'),rule=name,
                               calibration_firings=n,calibration_lates=positives,
                               calibration_nonlates=n-positives,
                               eligible_for_fitting=int(sufficient),
                               training_late_rate=float(y_cal[fired].mean()) if n else np.nan))
            if sufficient:supported.append(j)
        # Intercept-only shift is NOT claimed as knowledge effect. It is an explicit
        # reference candidate, and the gate must pass against the raw model.
        c=estimate_residuals(p_cal,y_cal,r_cal,supported)
        p_gate_candidate=predict(p_gate,r_gate,supported,c)
        raw_gate=brier_score_loss(y_gate,p_gate)
        candidate_gate=brier_score_loss(y_gate,p_gate_candidate)
        accepted=bool(candidate_gate+IMPROVEMENT_FLOOR < raw_gate and len(supported)>0)
        p_adaptive=predict(pt,rt,supported,c) if accepted else pt.copy()
        preds={'Raw XGBoost':pt,'Signed rule candidate':predict(pt,rt,supported,c),
               'Evidence-gated rule overlay':p_adaptive}
        for method,p in preds.items():
            rows.append(dict(test_month=month.strftime('%Y-%m'),method=method,
                             train_n=len(train),nested_calibration_n=cut,nested_gate_n=len(val)-cut,
                             test_n=len(test),test_lates=int(yt.sum()),prevalence=float(yt.mean()),
                             supported_rules='|'.join(RULES[j] for j in supported) or 'none',
                             gate_accepted=int(accepted),gate_raw_brier=raw_gate,
                             gate_candidate_brier=candidate_gate,intercept=float(c[0]),
                             **metrics(yt,p)))
        print(f"{month:%Y-%m}: {len(val)} validation ({cut} cal / {len(val)-cut} gate), "
              f"supported={[RULES[j] for j in supported]}, enabled={accepted}; "
              f"test AP raw={metrics(yt,pt)['AP']:.4f} gated={metrics(yt,p_adaptive)['AP']:.4f}; "
              f"Brier raw={metrics(yt,pt)['Brier']:.5f} gated={metrics(yt,p_adaptive)['Brier']:.5f}",flush=True)
    results=pd.DataFrame(rows)
    results.to_csv(out/'rule_reliability.csv',index=False,float_format='%.7f')
    pd.DataFrame(audits).to_csv(out/'rule_gate_audit.csv',index=False,float_format='%.7f')
    return results


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--data',type=Path,required=True)
    arg.add_argument('--out',type=Path,default=Path('results'))
    opts=arg.parse_args()
    opts.out.mkdir(parents=True,exist_ok=True)
    df=load_features(opts.data)
    result=run(df,opts.out)
    print('\nPR-AUC by month:')
    print(result.pivot(index='test_month',columns='method',values='AP').round(5).to_string())
    print('\nBrier by month:')
    print(result.pivot(index='test_month',columns='method',values='Brier').round(5).to_string())

if __name__=='__main__':
    main()