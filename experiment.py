"""Reproducible real-data Olist late-delivery experiment.

Usage: python experiment.py --data archive.zip --out results
No post-approval events or review scores are used as predictors.
Target: delivered calendar date strictly later than estimated delivery calendar date.
Cohort: completed deliveries only, meaning findings do not generalize to cancellations.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    f1_score, precision_score, recall_score, precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

SEED = 2026
BASE = [
    'estimated_window_days', 'approval_delay_hours', 'purchase_month',
    'purchase_dow', 'purchase_hour', 'items', 'distinct_sellers',
    'total_price', 'total_freight', 'total_weight_g', 'max_weight_g',
    'total_volume_cm3', 'same_state', 'geo_distance_km',
]
RULES = [
    'route_time_buffer_days', 'tight_route_commitment',
    'multi_seller_coordination', 'heavy_package_flag',
]


def read_table(z: ZipFile, filename: str) -> pd.DataFrame:
    return pd.read_csv(z.open(filename), low_memory=False)


def haversine_km(lat1, lng1, lat2, lng2):
    valid = np.isfinite(lat1) & np.isfinite(lng1) & np.isfinite(lat2) & np.isfinite(lng2)
    a, b, c, d = [np.radians(v) for v in (lat1, lng1, lat2, lng2)]
    ds = a - c
    dl = b - d
    h = np.sin(ds / 2)**2 + np.cos(a)*np.cos(c)*np.sin(dl / 2)**2
    return np.where(valid, 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))), np.nan)


def load_features(data_zip: Path) -> pd.DataFrame:
    with ZipFile(data_zip) as z:
        orders = read_table(z, 'olist_orders_dataset.csv')
        items = read_table(z, 'olist_order_items_dataset.csv')
        customers = read_table(z, 'olist_customers_dataset.csv')
        sellers = read_table(z, 'olist_sellers_dataset.csv')
        products = read_table(z, 'olist_products_dataset.csv')
        geo = read_table(z, 'olist_geolocation_dataset.csv')

    for k in ['order_purchase_timestamp','order_approved_at','order_delivered_customer_date','order_estimated_delivery_date']:
        orders[k] = pd.to_datetime(orders[k], errors='coerce')
    orders = orders.loc[(orders.order_status == 'delivered') & orders.order_approved_at.notna() &
                        orders.order_delivered_customer_date.notna() & orders.order_estimated_delivery_date.notna()].copy()
    # Preserve only orders for which seller, product, and item rows can be represented.
    items = items.merge(products[['product_id','product_weight_g','product_length_cm',
                                  'product_width_cm','product_height_cm']], on='product_id', how='left')
    items['volume'] = items.product_length_cm * items.product_width_cm * items.product_height_cm
    items = items.merge(sellers[['seller_id','seller_zip_code_prefix','seller_state']], on='seller_id', how='left')
    # Zip centroids are physical geography only; they do not depend on delivery outcomes.
    geo = geo.groupby('geolocation_zip_code_prefix', as_index=False)[['geolocation_lat','geolocation_lng']].median()
    seller_geo = geo.rename(columns={'geolocation_zip_code_prefix':'seller_zip_code_prefix',
                                     'geolocation_lat':'seller_lat', 'geolocation_lng':'seller_lng'})
    cust_geo = geo.rename(columns={'geolocation_zip_code_prefix':'customer_zip_code_prefix',
                                   'geolocation_lat':'customer_lat', 'geolocation_lng':'customer_lng'})
    items = items.merge(seller_geo, on='seller_zip_code_prefix', how='left')
    ag = items.groupby('order_id').agg(
        items=('order_item_id','size'), distinct_sellers=('seller_id','nunique'),
        total_price=('price','sum'),total_freight=('freight_value','sum'),
        total_weight_g=('product_weight_g','sum'), max_weight_g=('product_weight_g','max'),
        total_volume_cm3=('volume','sum'),seller_lat=('seller_lat','mean'),seller_lng=('seller_lng','mean'),
        seller_state=('seller_state','first'),
    ).reset_index()
    frame = orders.merge(customers[['customer_id','customer_state','customer_zip_code_prefix']],on='customer_id',how='inner')
    frame = frame.merge(ag, on='order_id',how='inner')
    frame = frame.merge(cust_geo,on='customer_zip_code_prefix',how='left')
    frame['geo_distance_km'] = haversine_km(frame.seller_lat.to_numpy(),frame.seller_lng.to_numpy(),
                                            frame.customer_lat.to_numpy(),frame.customer_lng.to_numpy())
    frame['late'] = (frame.order_delivered_customer_date.dt.normalize() >
                     frame.order_estimated_delivery_date.dt.normalize()).astype(int)
    frame['estimated_window_days'] = (frame.order_estimated_delivery_date-frame.order_approved_at).dt.total_seconds()/86400
    frame['approval_delay_hours'] = (frame.order_approved_at-frame.order_purchase_timestamp).dt.total_seconds()/3600
    frame['purchase_month'] = frame.order_purchase_timestamp.dt.month
    frame['purchase_dow'] = frame.order_purchase_timestamp.dt.dayofweek
    frame['purchase_hour'] = frame.order_purchase_timestamp.dt.hour
    frame['same_state'] = (frame.seller_state == frame.customer_state).astype(int)
    # Pre-specified illustrative operational rules. Constants are assumptions, not
    # observed delivery service-level agreements; test the contribution by ablation.
    frame['route_time_buffer_days'] = (frame.estimated_window_days -
                                        (1.0 + frame.geo_distance_km / 500.0 +
                                         (frame.distinct_sellers > 1).astype(float)))
    frame['tight_route_commitment'] = (frame.route_time_buffer_days < 0).astype(int)
    frame['multi_seller_coordination'] = (frame.distinct_sellers > 1).astype(int)
    frame['heavy_package_flag'] = (frame.max_weight_g > 10000).astype(int)
    frame = frame.sort_values(['order_purchase_timestamp','order_id']).reset_index(drop=True)
    assert frame.order_id.is_unique
    return frame


def score(y, p, threshold):
    yp = (p>=threshold).astype(int)
    return {'PR-AUC':average_precision_score(y,p), 'ROC-AUC':roc_auc_score(y,p),
            'Brier':brier_score_loss(y,p), 'Precision':precision_score(y,yp,zero_division=0),
            'Recall':recall_score(y,yp,zero_division=0),'F1':f1_score(y,yp,zero_division=0),
            'Alert rate':yp.mean(), 'Top 10% recall':y[np.argsort(p)[-max(1,int(len(p)*0.10)):]].sum()/y.sum()}


def best_threshold(y,p):
    precision, recall, thresholds = precision_recall_curve(y,p)
    f1s = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    # Prevent zero-alert or all-alert prevalence artefacts by using validation thresholds.
    return float(thresholds[int(np.argmax(f1s))]) if len(thresholds) else 0.5



def monthly_robustness(df, output_dir, n_boot=500):
    """Strictly chronological monthly tests, with outcomes known at fitting time.

    Train: orders purchased before the month preceding test month, and delivered
    before that preceding month's start (label availability at model fit).
    Validate: preceding month's orders already delivered by test-month start.
    Test: test-month orders, labelled after delivery solely for evaluation.
    No tuning using the test window; all model hyperparameters are fixed.
    Order-level bootstrap intervals are descriptive (repeated customers possible).
    """
    rng = np.random.default_rng(SEED)
    fold_rows = []
    months = pd.to_datetime(['2018-03-01','2018-04-01','2018-05-01',
                             '2018-06-01','2018-07-01'])
    for cutoff in months:
        val_begin = cutoff - pd.DateOffset(months=1)
        test_end = cutoff + pd.DateOffset(months=1)
        tr = df.loc[(df.order_purchase_timestamp < val_begin) &
                    (df.order_delivered_customer_date < val_begin)].copy()
        va = df.loc[(df.order_purchase_timestamp >= val_begin) &
                    (df.order_purchase_timestamp < cutoff) &
                    (df.order_delivered_customer_date < cutoff)].copy()
        te = df.loc[(df.order_purchase_timestamp >= cutoff) &
                    (df.order_purchase_timestamp < test_end)].copy()
        assert tr.order_purchase_timestamp.max() < val_begin
        assert tr.order_delivered_customer_date.max() < val_begin
        assert va.order_delivered_customer_date.max() < cutoff
        assert te.order_purchase_timestamp.min() >= cutoff
        if len(tr) < 1000 or len(va) < 100 or te.late.sum() < 15:
            raise RuntimeError(f'Insufficient observations for robustness test {cutoff}')
        ytr, yva, yte = tr.late.to_numpy(),va.late.to_numpy(),te.late.to_numpy()
        preds={}
        for model_name, cols, model in [
            ('Rule-only', RULES,make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=1200,random_state=SEED))),
            ('Logistic', BASE,make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=1200,random_state=SEED))),
            ('XGBoost',BASE,XGBClassifier(n_estimators=260,max_depth=3,
                     learning_rate=.045,subsample=.85,colsample_bytree=.9,
                     objective='binary:logistic',eval_metric='logloss',
                     tree_method='hist',n_jobs=4,random_state=SEED)),
            ('Knowledge-guided XGBoost',BASE+RULES,XGBClassifier(
                     n_estimators=260,max_depth=3,learning_rate=.045,
                     subsample=.85,colsample_bytree=.9,objective='binary:logistic',
                     eval_metric='logloss',tree_method='hist',n_jobs=4,random_state=SEED)),
        ]:
            imp=SimpleImputer(strategy='median')
            xtr=imp.fit_transform(tr[cols]); xva=imp.transform(va[cols]); xte=imp.transform(te[cols]);
            model.fit(xtr,ytr)
            pv=model.predict_proba(xva)[:,1]
            pt=model.predict_proba(xte)[:,1]
            threshold=best_threshold(yva,pv)
            preds[model_name]=pt
            sc=score(yte,pt,threshold)
            fold_rows.append(dict(fold=cutoff.strftime('%Y-%m'),model=model_name,
                    train_n=len(tr),validation_n=len(va),test_n=len(te),
                    test_positives=int(yte.sum()),test_prevalence=float(yte.mean()),
                    **sc))
        # CI around a paired difference in PR-AUC within each heldout month.
        indices=np.arange(len(yte))
        for alternative in ['Rule-only','Knowledge-guided XGBoost']:
            diffs=[]
            for _ in range(n_boot):
                ind=rng.choice(indices, size=len(indices),replace=True)
                yy=yte[ind]
                if len(np.unique(yy)) != 2:
                    continue
                diffs.append(average_precision_score(yy,preds[alternative][ind])-
                             average_precision_score(yy,preds['XGBoost'][ind]))
            lower,upper=np.quantile(diffs,[.025,.975])
            for record in fold_rows[-4:]:
                if record['model']==alternative:
                    record['PR-AUC delta vs XGBoost']=average_precision_score(yte,preds[alternative])-average_precision_score(yte,preds['XGBoost'])
                    record['Paired bootstrap lower']=lower
                    record['Paired bootstrap upper']=upper
    out=pd.DataFrame(fold_rows)
    out.to_csv(output_dir/'rolling_validation.csv',index=False,float_format='%.6f')
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('results'))
    parser.add_argument('--rolling',action='store_true',help='Run five additional chronological monthly validation folds')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    df=load_features(args.data)
    if args.rolling:
        folds=monthly_robustness(df,args.out)
        print("\nAdditional out-of-time validation:\n",folds.pivot(index="fold",columns="model",values="PR-AUC").round(4).to_string())
        print("\nStored rolling_validation.csv. No additional figures or result tables created.")
        return
    n=len(df); i_val=int(n*.70); i_test=int(n*.85)
    val_start=df.order_purchase_timestamp.iloc[i_val]
    test_start=df.order_purchase_timestamp.iloc[i_test]
    # Purge training/validation records whose real outcomes were unavailable at
    # the moment the later evaluation period began.
    tr=df.iloc[:i_val].loc[lambda d:d.order_delivered_customer_date<val_start].copy()
    va=df.iloc[i_val:i_test].loc[lambda d:d.order_delivered_customer_date<test_start].copy()
    te=df.iloc[i_test:].copy()
    print('Cohort total',n,'late',round(df.late.mean(),4))
    print('Train/val/test',len(tr),len(va),len(te),'rates',*(round(x.late.mean(),4) for x in [tr,va,te]))
    print('Chronological cutoffs',val_start,test_start)
    stats = {'cohort':n,'late_rate':float(df.late.mean()),'train':len(tr),'validation':len(va),
             'test':len(te),'train_late_rate':float(tr.late.mean()),'val_late_rate':float(va.late.mean()),
             'test_late_rate':float(te.late.mean()),'validation_start':str(val_start),'test_start':str(test_start),
             'unobserved_geo_rate':float(df.geo_distance_km.isna().mean()),'dataset':'Olist Brazil e-commerce'}
    pd.Series(stats).to_csv(args.out/'sample_audit.csv',header=['value'])
    ytr=tr.late.to_numpy(); yva=va.late.to_numpy(); yte=te.late.to_numpy()
    # Median imputation fitted exclusively to pre-validation training records.
    imputer=SimpleImputer(strategy='median',add_indicator=False)
    xtr=imputer.fit_transform(tr[BASE]); xva=imputer.transform(va[BASE]); xte=imputer.transform(te[BASE])
    imp_kn=SimpleImputer(strategy='median')
    xtr_kn=imp_kn.fit_transform(tr[BASE+RULES]); xva_kn=imp_kn.transform(va[BASE+RULES]); xte_kn=imp_kn.transform(te[BASE+RULES])
    # Knowledge-only ablation uses only business rule signals, no base covariates.
    imp_rules=SimpleImputer(strategy='median')
    xrtr=imp_rules.fit_transform(tr[RULES]); xrva=imp_rules.transform(va[RULES]); xrte=imp_rules.transform(te[RULES])
    models={
        'Prior-only':(DummyClassifier(strategy='prior'),xtr,xva,xte),
        'Logistic':(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1200,random_state=SEED)),xtr,xva,xte),
        'Rule-only':(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1200,random_state=SEED)),xrtr,xrva,xrte),
        'XGBoost':(XGBClassifier(n_estimators=260,max_depth=3,learning_rate=.045,subsample=.85,
                   colsample_bytree=.9,objective='binary:logistic',eval_metric='logloss',
                   tree_method='hist',n_jobs=4,random_state=SEED),xtr,xva,xte),
        'Knowledge-guided XGBoost':(XGBClassifier(n_estimators=260,max_depth=3,learning_rate=.045,subsample=.85,
                   colsample_bytree=.9,objective='binary:logistic',eval_metric='logloss',
                   tree_method='hist',n_jobs=4,random_state=SEED),xtr_kn,xva_kn,xte_kn),
    }
    rows=[]; probabilities={}; thresholds={}
    for name,(model,xxtr,xxva,xxte) in models.items():
        model.fit(xxtr,ytr)
        pv=model.predict_proba(xxva)[:,1]
        pt=model.predict_proba(xxte)[:,1]
        threshold=best_threshold(yva,pv) if name!='Prior-only' else .5
        thresholds[name]=threshold
        s=score(yte,pt,threshold)
        rows.append({'Model':name,**{k:round(v,4) for k,v in s.items()},'Validation threshold':round(threshold,4)})
        probabilities[name]=pt
    metrics=pd.DataFrame(rows)
    metrics.to_csv(args.out/'table1_model_results.csv',index=False)
    base=metrics.loc[metrics.Model=='XGBoost'].iloc[0]
    full=metrics.loc[metrics.Model=='Knowledge-guided XGBoost'].iloc[0]
    only=metrics.loc[metrics.Model=='Rule-only'].iloc[0]
    ablation=pd.DataFrame([
        {'Experiment':'Knowledge signals alone', 'Variant':'Rule-only', 'PR-AUC':only['PR-AUC'],'ROC-AUC':only['ROC-AUC'],'Brier':only['Brier']},
        {'Experiment':'Knowledge-feature ablation', 'Variant':'XGBoost without rules','PR-AUC':base['PR-AUC'],'ROC-AUC':base['ROC-AUC'],'Brier':base['Brier']},
        {'Experiment':'Full system', 'Variant':'XGBoost with rules', 'PR-AUC':full['PR-AUC'],'ROC-AUC':full['ROC-AUC'],'Brier':full['Brier']},
        {'Experiment':'Incremental gain (full - ablated)', 'Variant':'Delta',
         'PR-AUC':round(full['PR-AUC']-base['PR-AUC'],4),
         'ROC-AUC':round(full['ROC-AUC']-base['ROC-AUC'],4),
         'Brier':round(full['Brier']-base['Brier'],4)},
    ])
    ablation.to_csv(args.out/'table2_knowledge_ablation.csv',index=False)
    plt.rcParams.update({'font.size':11,'figure.dpi':160,'savefig.dpi':240})
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    for name in ['Logistic','XGBoost','Knowledge-guided XGBoost','Rule-only']:
        pr,rec,_=precision_recall_curve(yte,probabilities[name]); ax.plot(rec,pr,lw=1.9,label=f'{name}  AP={average_precision_score(yte,probabilities[name]):.3f}')
    ax.axhline(yte.mean(),linestyle=':',color='gray',label=f'Test prevalence={yte.mean():.3f}')
    ax.set(xlabel='Recall of late deliveries',ylabel='Precision',title='Figure 1. Out-of-time precision-recall performance')
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8)
    fig.tight_layout();fig.savefig(args.out/'figure1_precision_recall.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,4.8));ax.plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
    for name in ['Logistic','XGBoost','Knowledge-guided XGBoost']:
        fraction,means=calibration_curve(yte,probabilities[name],n_bins=10,strategy='quantile')
        ax.plot(means,fraction,'o-',lw=1.6,ms=4,label=f'{name}  Brier={brier_score_loss(yte,probabilities[name]):.3f}')
    ax.set(xlabel='Mean predicted probability',ylabel='Observed late-delivery frequency',title='Figure 2. Out-of-time calibration (10 quantile bins)')
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend(loc='best',fontsize=8)
    fig.tight_layout();fig.savefig(args.out/'figure2_calibration.png');plt.close(fig)
    print('\n',metrics.to_string(index=False));print('\n',ablation.to_string(index=False))
    print('\nWrote exactly two scientific figures and two experimental tables, plus sample_audit.csv.')

if __name__=='__main__':
    main()