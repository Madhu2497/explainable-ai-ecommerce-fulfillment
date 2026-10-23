"""Exploratory knowledge-constrained decision validation for the Olist study.

Reproduces a genuinely constrained monotone rule layer over baseline XGBoost:
    logit(q) = a + b * logit(p_xgb) + sum_j w_j * r_j
where b >= 0, w_j >= 0; fitted on prior-month observable labels only.
The 3 rule signs are PRE-SPECIFIED HYPOTHESES, not verified service-level rules.
Baseline Platt calibrator uses the same logistic link without rules.

Usage: python decision_validation.py --data archive.zip --out results
Writes supplementary CSVs; manuscript still requires only two MAIN tables/figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier
from experiment import load_features, BASE, SEED

RULES = ['tight_route_commitment','multi_seller_coordination','heavy_package_flag']
MONTHS = pd.to_datetime(['2018-03-01','2018-04-01','2018-05-01','2018-06-01','2018-07-01'])
REGULARIZATION = 0.02  # fixed before running; not selected using test outcomes
ALERT_BUDGET = 0.10


def predictor():
    return XGBClassifier(n_estimators=260,max_depth=3,learning_rate=.045,
                         subsample=.85,colsample_bytree=.9,objective='binary:logistic',
                         eval_metric='logloss',tree_method='hist',n_jobs=4,random_state=SEED)


def correction_fit(p_val, y_val, rule_val, *, with_rules:bool):
    """Regularized nonnegative-slope logistic residual calibration.

    Validation-only fit. Nonnegative rule effects represent expert hypotheses;
    an estimated 0 coefficient means the model ignores an unsupported rule.
    """
    pv = np.clip(np.asarray(p_val, dtype=float),1e-5,1-1e-5)
    y = np.asarray(y_val, dtype=float)
    z = logit(pv)
    R = np.asarray(rule_val, dtype=float)
    if not with_rules:
        R=R[:, :0]
    mat = np.column_stack((np.ones(len(z)), z, R))

    def objective(params):
        q=np.clip(expit(mat @ params),1e-9,1-1e-9)
        loss=-np.mean(y*np.log(q)+(1-y)*np.log(1-q))
        # Strong regularization to prevent fitting transient labels in one month.
        penalty=REGULARIZATION*((params[1]-1.0)**2+np.dot(params[2:],params[2:]))
        grad=mat.T @ (q-y)/len(y)
        rg=np.zeros_like(params)
        rg[1]=2*REGULARIZATION*(params[1]-1.0)
        rg[2:]=2*REGULARIZATION*params[2:]
        return loss+penalty, grad+rg

    init = np.r_[0.0,1.0,np.zeros(R.shape[1])]
    bounds = [(-10.0, 10.0),(0.05,3.0)] + [(0.0,3.0)]*R.shape[1]
    fit = minimize(objective, init, jac=True, method='L-BFGS-B',bounds=bounds,
                   options={'maxiter':350,'ftol':1e-12})
    if not fit.success and np.linalg.norm(fit.jac)>1e-3:
        raise RuntimeError(f'Calibration optimization failed: {fit.message}')
    return fit.x


def apply_correction(p, rules, coefficients):
    v=np.clip(np.asarray(p, dtype=float),1e-5,1-1e-5)
    R=np.asarray(rules, dtype=float)
    return expit(coefficients[0]+coefficients[1]*logit(v)+
                 (R[:, :len(coefficients)-2] @ coefficients[2:]
                  if len(coefficients)>2 else 0))


def topk_metrics(y, p, fraction=ALERT_BUDGET):
    """Deterministic fixed-capacity prospective *screening* proxy, not actual uplift."""
    n=len(y); k=max(1,round(n*fraction))
    top=np.argsort(-np.asarray(p),kind='mergesort')[:k]
    positives=y.sum(); tp=y[top].sum()
    # Proxy only: illustrative cost of missed late (5) and false alert (1).
    # Does not measure intervention efficacy or financial savings.
    proxy_cost=5.0*(positives-tp)+1.0*(k-tp)
    return k, float(tp/k),float(tp/positives),float(proxy_cost/n)


def expected_calibration_error(y,p,nbins=10):
    bins=np.minimum((np.clip(p,0,1)*nbins).astype(int),nbins-1)
    return float(sum(np.sum(bins==b)*abs(np.mean(y[bins==b])-np.mean(p[bins==b]))
                     for b in range(nbins) if np.any(bins==b))/len(y))


def metrics(y,p):
    k,precision,recall,cost=topk_metrics(y,p)
    return {'AP':float(average_precision_score(y,p)),
            'ROC_AUC':float(roc_auc_score(y,p)),
            'Brier':float(brier_score_loss(y,p)),
            'LogLoss':float(log_loss(y,p,labels=[0,1])),
            'ECE_10':expected_calibration_error(y,p),
            'alert_count':k,'precision_at_10pct':precision,
            'recall_at_10pct':recall,'illustrative_screening_cost_per_order':cost}


def paired_bootstrap(y,p_alt,p_ref, *, nboot=300, seed=SEED):
    """Exploratory paired order-level resampling; not cluster-robust or independent months."""
    rng=np.random.default_rng(seed)
    n=len(y)
    diffs=[]
    for _ in range(nboot):
        idx=rng.integers(0,n,size=n)
        yy=y[idx]
        if yy.sum()==0 or yy.sum()==n:
            continue
        diffs.append((brier_score_loss(yy,p_alt[idx])-brier_score_loss(yy,p_ref[idx]),
                      topk_metrics(yy,p_alt[idx])[2]-topk_metrics(yy,p_ref[idx])[2]))
    a=np.asarray(diffs)
    return np.quantile(a[:,0],[.025,.975]).tolist(),np.quantile(a[:,1],[.025,.975]).tolist()


def run(df, out):
    rows=[]
    for month in MONTHS:
        val_begin=month-pd.DateOffset(months=1)
        test_end=month+pd.DateOffset(months=1)
        train=df.loc[(df.order_purchase_timestamp<val_begin)&
                     (df.order_delivered_customer_date<val_begin)]
        valid=df.loc[(df.order_purchase_timestamp>=val_begin)&
                     (df.order_purchase_timestamp<month)&
                     (df.order_delivered_customer_date<month)]
        test=df.loc[(df.order_purchase_timestamp>=month)&
                    (df.order_purchase_timestamp<test_end)]
        if min(len(train),len(valid),len(test))<100 or test.late.sum()<10:
            raise RuntimeError('Insufficient observations in '+str(month))
        assert train.order_delivered_customer_date.max()<val_begin
        assert valid.order_delivered_customer_date.max()<month
        assert test.order_purchase_timestamp.min()>=month
        imp=SimpleImputer(strategy='median')
        Xtrain=imp.fit_transform(train[BASE]); Xvalid=imp.transform(valid[BASE]); Xtest=imp.transform(test[BASE])
        ytrain=train.late.to_numpy(); yvalid=valid.late.to_numpy(); ytest=test.late.to_numpy()
        mdl=predictor(); mdl.fit(Xtrain,ytrain)
        pv=mdl.predict_proba(Xvalid)[:,1]; pt=mdl.predict_proba(Xtest)[:,1]
        rv=valid[RULES].fillna(0).to_numpy(dtype=float)
        rt=test[RULES].fillna(0).to_numpy(dtype=float)
        alpha=correction_fit(pv,yvalid,rv,with_rules=False)
        beta=correction_fit(pv,yvalid,rv,with_rules=True)
        preds={'Raw XGBoost':pt,
               'Global calibrated XGBoost':apply_correction(pt,rt,alpha),
               'Knowledge-constrained calibration':apply_correction(pt,rt,beta)}
        # Control: use only domain-rule risk rank (no test-learned parameters)
        # This is distinct from the rule-only logistic baseline in experiment.py.
        preds['Fixed rule-priority']=expit(0.8*rt[:,0]+0.5*rt[:,1]+0.4*rt[:,2]-2.0)
        for method,pred in preds.items():
            sc=metrics(ytest,pred)
            row={'month':month.strftime('%Y-%m'), 'method':method, 'train_n':len(train),
                 'validation_n':len(valid),'test_n':len(test),'test_lates':int(ytest.sum()),
                 'test_prevalence':float(ytest.mean()), **sc}
            if method=='Knowledge-constrained calibration':
                row.update({'intercept':float(beta[0]), 'logit_slope':float(beta[1]),
                            'weight_tight':float(beta[2]), 'weight_multiseller':float(beta[3]),
                            'weight_heavy':float(beta[4])})
                ci_brier,ci_recall=paired_bootstrap(ytest,pred,pt,seed=SEED+int(month.month))
                row.update({'delta_Brier_vs_raw':sc['Brier']-metrics(ytest,pt)['Brier'],
                            'delta_Recall10_vs_raw':sc['recall_at_10pct']-metrics(ytest,pt)['recall_at_10pct'],
                            'delta_Brier_CI_low':ci_brier[0], 'delta_Brier_CI_high':ci_brier[1],
                            'delta_Recall10_CI_low':ci_recall[0], 'delta_Recall10_CI_high':ci_recall[1]})
            rows.append(row)
        print(f"{month.strftime('%Y-%m')} test={len(test)} positives={ytest.sum()} "
              f"AP raw={metrics(ytest,pt)['AP']:.4f} "
              f"constrained={metrics(ytest,preds['Knowledge-constrained calibration'])['AP']:.4f} "
              f"Brier raw={metrics(ytest,pt)['Brier']:.4f} "
              f"constrained={metrics(ytest,preds['Knowledge-constrained calibration'])['Brier']:.4f}", flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'decision_validation.csv',index=False,float_format='%.7f')
    return frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('results'))
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    df=load_features(args.data)
    # Full-cohort DESCRIPTIVE rule coverage only; never used to tune algorithms.
    diagnostics=[]
    for name in RULES:
        for flag in (0,1):
            group=df.loc[df[name]==flag]
            diagnostics.append({'rule':name, 'value':flag,'orders':len(group),
                                'fraction_of_cohort':len(group)/len(df),
                                'late_rate':float(group.late.mean())})
    pd.DataFrame(diagnostics).to_csv(args.out/'rule_coverage_audit.csv',index=False,float_format='%.7f')
    frame=run(df,args.out)
    print('Finished; decision_validation.csv includes only real-data validation.')
    print('\nBrier by month:\n',frame.pivot(index='month',columns='method',values='Brier').round(4))
    print('\nRecall @ 10% alerts:\n',frame.pivot(index='month',columns='method',values='recall_at_10pct').round(4))

if __name__=='__main__':
    main()