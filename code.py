# code.py

import sys
import math
import random
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


CURRENCY = "₸"
TRAVEL_CASHBACK = 0.04
PREMIUM_TIERS = {"low": 0.02, "mid": 0.035, "high": 0.045}
PREMIUM_THRESH = {"mid": 300_000, "high": 1_500_000}
CREDIT_CARD_RATE = 0.10
FX_RATE = 0.005
ATM_FEE = 300
MIN_BALANCE_INVEST = 50_000
DEPOSIT_RATE_MULT = 0.005
SAVING_ACC_RATE = 0.01
SAVING_FREEZE_RATE = 0.02

RANDOM_STATE = 42
MAX_PUSH_LEN = 220

PRODUCTS = [
    'Карта для путешествий',
    'Премиальная карта',
    'Кредитная карта',
    'Обмен валют',
    'Кредит наличными',
    'Депозит мультивалютный',
    'Вклад сберегательный',
    'Вклад накопительный',
    'Инвестиции (брокерский счёт)',
    'Золотые слитки'
]


RU_MONTHS = {
    1: "январе", 2: "феврале", 3: "марте", 4: "апреле", 5: "мае", 6: "июне",
    7: "июле", 8: "августе", 9: "сентябре", 10: "октябре", 11: "ноябре", 12: "декабре"
}


CTAS = ["Оформить", "Узнать лимит", "Открыть вклад", "Подключить", "Настроить", "Купить"]
TONES = ["дружелюбный", "формальный", "экспертный"]


def ru_month_name(dt=None):
    dt = dt or datetime.now()
    return RU_MONTHS[dt.month]

def format_amount(amount):
    try:
        a = float(amount)
    except:
        a = 0.0
    a = int(round(a / 100.0) * 100)
    s = f"{a:,}".replace(",", " ")
    return f"{s} {CURRENCY}"

def safe_read_csv(path, parse_dates=None):
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        raise RuntimeError(f"Failed reading {path}: {e}")

def top_categories(tx_client, n=3):
    if tx_client.empty or 'category' not in tx_client.columns:
        return []
    sums = tx_client.groupby('category')['amount'].sum()
    return list(sums.sort_values(ascending=False).head(n).index)

def sum_categories(tx_client, cats):
    if tx_client.empty or 'category' not in tx_client.columns:
        return 0.0
    return float(tx_client[tx_client['category'].isin(cats)]['amount'].sum())

def sum_currency(tx_client, currencies):
    if tx_client.empty or 'currency' not in tx_client.columns:
        return 0.0
    return float(tx_client[tx_client['currency'].isin(currencies)]['amount'].sum())

def inflow_outflow_ratio(transfers_client, tx_client):
    if transfers_client is None or transfers_client.empty:
        inflow = 0.0
        outflow = 0.0
    else:
        inflow = float(transfers_client.loc[transfers_client['direction']=="in",'amount'].sum())
        outflow = float(transfers_client.loc[transfers_client['direction']=="out",'amount'].sum())
    outflow += float(tx_client['amount'].sum()) if not tx_client.empty else 0.0
    return (inflow + 1.0) / (outflow + 1.0)

def monthly_volatility(tx_client):
    if tx_client.empty or 'date' not in tx_client.columns:
        return 0.0
    t = tx_client.copy()
    t['month'] = t['date'].dt.to_period('M')
    monthly = t.groupby('month')['amount'].sum()
    if len(monthly) < 2:
        return 0.0
    return float(monthly.std() / (monthly.mean() + 1e-9))

def detect_regular_topups(transfers_client, window_months=3, threshold=3):
    if transfers_client is None or transfers_client.empty or 'date' not in transfers_client.columns:
        return False
    recent = transfers_client[transfers_client['direction']=='in'].copy()
    if recent.empty:
        return False
    recent['month'] = recent['date'].dt.to_period('M')
    return int(recent['month'].nunique()) >= threshold
def count_ops_by_type(transfers_client, pattern):
    if transfers_client is None or transfers_client.empty:
        return 0
    return int(transfers_client['type'].str.contains(pattern, na=False).sum())


def benefit_travel(profile, tx_client, transfers_client):
    cats = ['Путешествия','Отели','Такси','Авиабилеты','Transport']
    spend = sum_categories(tx_client, cats)
    trips = int(tx_client[tx_client['category'].isin(cats)].shape[0]) if not tx_client.empty else 0
    benefit = TRAVEL_CASHBACK * spend
    return float(benefit), {'spend': spend, 'n_trips': trips}

def benefit_premium(profile, tx_client, transfers_client):
    balance = float(profile.get('avg_monthly_balance_KZT', 0.0) or 0.0)
    if balance >= PREMIUM_THRESH['high']:
        tier = 'high'
    elif balance >= PREMIUM_THRESH['mid']:
        tier = 'mid'
    else:
        tier = 'low'
    base_spend = float(tx_client['amount'].sum()) if not tx_client.empty else 0.0
    special = sum_categories(tx_client, ['Кафе и рестораны','Ювелирные украшения','Косметика и Парфюмерия'])
    atm_count = count_ops_by_type(transfers_client, 'atm_withdrawal')
    fees = atm_count * ATM_FEE
    benefit = PREMIUM_TIERS[tier] * base_spend + 0.03 * special + fees
    return float(benefit), {'balance': balance, 'tier': tier, 'atm_count': atm_count}

def benefit_credit(profile, tx_client, transfers_client):
    top3 = top_categories(tx_client, 3)
    top_spend = sum_categories(tx_client, top3)
    online = sum_categories(tx_client, ['Интернет','Онлайн покупки','E-commerce','Online'])
    benefit = CREDIT_CARD_RATE * (top_spend + online)
    if count_ops_by_type(transfers_client, 'installment') > 0:
        benefit *= 1.15
    return float(benefit), {'cats': top3, 'online_spend': online}

def benefit_fx(profile, tx_client, transfers_client):
    fx_ops = count_ops_by_type(transfers_client, 'fx')
    fx_sum = sum_currency(tx_client, ['USD','EUR'])
    benefit = FX_RATE * fx_sum + fx_ops * 200.0
    return float(benefit), {'fx_ops': fx_ops, 'fx_sum': fx_sum}

def benefit_cash(profile, tx_client, transfers_client):
    ratio = inflow_outflow_ratio(transfers_client, tx_client)
    balance = float(profile.get('avg_monthly_balance_KZT', 0.0) or 0.0)
    vol = monthly_volatility(tx_client)
    if (balance < 100_000 and ratio < 0.7) or vol > 0.5:
        est = min(200_000, int((100_000 - balance) * 0.5 + 50_000))
        benefit = float(max(0, est))
    else:
        benefit = 0.0
    return float(benefit), {'ratio': ratio, 'balance': balance, 'volatility': vol}

def benefit_deposit_multi(profile, tx_client, transfers_client):
    balance = float(profile.get('avg_monthly_balance_KZT', 0.0) or 0.0)
    fx_sum = sum_currency(tx_client, ['USD','EUR'])
    benefit = DEPOSIT_RATE_MULT * balance + 0.002 * fx_sum
    return float(benefit), {'balance': balance, 'fx_sum': fx_sum}

def benefit_saving_freeze(profile, tx_client, transfers_client):
    balance = float(profile.get('avg_monthly_balance_KZT', 0.0) or 0.0)
    vol = monthly_volatility(tx_client)
    regular = detect_regular_topups(transfers_client)
    if balance > 300_000 and vol < 0.25 and regular:
        benefit = SAVING_FREEZE_RATE * balance
    else:
        benefit = 0.0
    return float(benefit), {'balance': balance, 'vol': vol, 'regular_topups': regular}

def benefit_saving_acc(profile, tx_client, transfers_client):
    balance = float(profile.get('avg_monthly_balance_KZT', 0.0) or 0.0)
    regular = detect_regular_topups(transfers_client)
    if 30_000 <= balance <= 400_000 and regular:
        benefit = SAVING_ACC_RATE * balance
    else:
        benefit = 0.0
    return float(benefit), {'balance': balance, 'regular_topups': regular}

def benefit_invest(profile, tx_client, transfers_client):
    balance = float(profile.get('avg_monthly_balance_KZT', 0.0) or 0.0)
    if balance >= MIN_BALANCE_INVEST:
        benefit = 0.006 * balance
    else:
        benefit = 0.0
    return float(benefit), {'balance': balance}
def benefit_gold(profile, tx_client, transfers_client):
    jew = sum_categories(tx_client, ['Ювелирные украшения','Золото'])
    ops = count_ops_by_type(transfers_client, 'gold')
    benefit = 0.002 * jew + ops * 5000.0
    return float(benefit), {'jew': jew, 'ops': ops}

BENEFIT_FUNCS = [
    benefit_travel,
    benefit_premium,
    benefit_credit,
    benefit_fx,
    benefit_cash,
    benefit_deposit_multi,
    benefit_saving_freeze,
    benefit_saving_acc,
    benefit_invest,
    benefit_gold
]


PUSH_TEMPLATES = {
    'Карта для путешествий': [
        "{name}, в {month} вы потратили {sum_travel} на поездки/таксы/отели — с тревел-картой вернули бы ≈{benefit}. {cta}",
        "{name}, частые поездки? Тревел-карта вернёт часть расходов — ≈{benefit}. {cta}"
    ],
    'Премиальная карта': [
        "{name}, ваш средний остаток {balance}. Премиальная карта даст повышенный кешбэк и бесплатные снятия — ≈{benefit}. {cta}",
        "{name}, хотите больше кешбэка и бесплатные снятия? Премиальная карта — ≈{benefit}. {cta}"
    ],
    'Кредитная карта': [
        "{name}, ваши топ-категории: {cats}. Кредитка даст бонусы до 10% в любимых категориях. {cta}",
        "{name}, выгодные покупки в {cats} — кредитная карта даст бонусы. {cta}"
    ],
    'Обмен валют': [
        "{name}, вы тратите {fx_sum} в валюте. Авто-обмен и выгодный курс помогут сэкономить ≈{benefit}. {cta}",
    ],
    'Кредит наличными': [
        "{name}, нужна подушка? Возможный лимит ~{benefit}. {cta}",
    ],
    'Депозит мультивалютный': [
        "{name}, у вас валютные остатки {fx_sum}. Разместите их во вкладе — получите доход. {cta}",
    ],
    'Вклад сберегательный': [
        "{name}, баланс {balance}. Сберегательный вклад принесёт доход ≈{benefit}. {cta}",
    ],
    'Вклад накопительный': [
        "{name}, регулярно пополняете? Накопительный вклад поможет копить с повышенной ставкой. {cta}",
    ],
    'Инвестиции (брокерский счёт)': [
        "{name}, начните инвестировать с малого — порог низкий и есть рекомендации. {cta}",
        "{name}, хотите попробовать инвестиции? Низкий порог входа и персональные рекомендации. {cta}"
    ],
    'Золотые слитки': [
        "{name}, частые покупки ювелирки ({jew}). Рассмотрите золото как диверсификацию. {cta}",
    ]
}

def choose_cta_and_tone(product, profile):
    
    balance = float(profile.get('avg_monthly_balance_KZT', 0.0) or 0.0)
    age = profile.get('age', None)
    tone = random.choice(TONES)
    if balance > 1_000_000:
        tone = "экспертный"
    elif age is not None and age < 25:
        tone = "дружелюбный"
    cta = random.choice(CTAS)
    return cta, tone

def generate_push_text(profile, product, meta):
    templates = PUSH_TEMPLATES.get(product, ["{name}, у нас есть персональное предложение. {cta}"])
    tpl = random.choice(templates)
    cta, tone = choose_cta_and_tone(product, profile)
   
    name = str(profile.get('name') or profile.get('client_code') or "Клиент")
    month = ru_month_name()
    balance = format_amount(profile.get('avg_monthly_balance_KZT', 0.0))
    benefit = format_amount(meta.get('benefit_est', meta.get('benefit', 0.0)))
    fx_sum = format_amount(meta.get('fx_sum', 0.0)) if meta.get('fx_sum', 0.0) else "0"
    sum_travel = format_amount(meta.get('spend', 0.0)) if meta.get('spend', 0.0) else "0"
    cats = ", ".join(meta.get('cats', [])) if meta.get('cats') else ""
    jew = format_amount(meta.get('jew', 0.0)) if meta.get('jew', 0.0) else "0"
    text = tpl.format(name=name, month=month, balance=balance, benefit=benefit,
                      fx_sum=fx_sum, sum_travel=sum_travel, cats=cats, jew=jew, cta=cta)
   
    if len(text) > MAX_PUSH_LEN:
        text = text[:MAX_PUSH_LEN-3].rsplit(" ",1)[0] + "..."
    return text

def extract_features_for_client(profile_row, tx_df, transfers_df):
    # tx_df and transfers_df already filtered for client
    tx = tx_df if tx_df is not None else pd.DataFrame()
    transfers = transfers_df if transfers_df is not None else pd.DataFrame()
    features = {}
    # basic numeric features
    features['avg_balance'] = float(profile_row.get('avg_monthly_balance_KZT', 0.0) or 0.0)
    features['total_spend'] = float(tx['amount'].sum()) if not tx.empty else 0.0
    features['num_transactions'] = int(tx.shape[0]) if not tx.empty else 0
    features['monthly_volatility'] = monthly_volatility(tx)
    features['inflow_outflow_ratio'] = inflow_outflow_ratio(transfers, tx)
    features['regular_topups'] = int(detect_regular_topups(transfers))
    features['atm_ops'] = int(count_ops_by_type(transfers, 'atm_withdrawal'))
    features['installment_ops'] = int(count_ops_by_type(transfers, 'installment'))
    features['fx_ops'] = int(count_ops_by_type(transfers, 'fx'))
    features['fx_spend'] = float(sum_currency(tx, ['USD', 'EUR']))
    features['jew_spend'] = float(sum_categories(tx, ['Ювелирные украшения', 'Золото']))
    top_cats = top_categories(tx, 5)
    for i,cat in enumerate(top_cats):
        features[f'top_cat_{i+1}'] = cat
    
    for i in range(len(top_cats),5):
        features[f'top_cat_{i+1}'] = ""
    return features


def rule_based_top4_and_meta(profile_row, tx_df, transfers_df):
    
    benefits = []
    for name, func in zip(PRODUCTS, BENEFIT_FUNCS):
        b, meta = func(profile_row, tx_df, transfers_df)
        meta = dict(meta)
        meta['product'] = name
        meta['benefit_est'] = float(b)
        benefits.append((name, float(b), meta))
    benefits_sorted = sorted(benefits, key=lambda x: x[1], reverse=True)
    top4 = benefits_sorted[:4]
    return top4, benefits_sorted


def build_feature_matrix_and_labels(profiles_df, tx_df, transfers_df):
    rows = []
    for _, p in profiles_df.iterrows():
        client_code = p['client_code']
        tx_c = tx_df[tx_df['client_code'] == client_code] if not tx_df.empty else pd.DataFrame()
        tr_c = transfers_df[transfers_df['client_code'] == client_code] if not transfers_df.empty else pd.DataFrame()
        feats = extract_features_for_client(p, tx_c, tr_c)
        top4, all_benefits = rule_based_top4_and_meta(p, tx_c, tr_c)        
        label = top4[0][0] if top4 else None
        flat = feats.copy()
        flat['client_code'] = int(client_code)
        flat['label'] = label
        rows.append(flat)
    df = pd.DataFrame(rows)
    df.fillna("", inplace=True)
    cat_columns = [f'top_cat_{i+1}' for i in range(5)]
    df_encoded = pd.get_dummies(df, columns=cat_columns, dummy_na=False, drop_first=False)
    return df_encoded

def train_ml_model(df_encoded):
    if 'label' not in df_encoded.columns:
        raise RuntimeError("Labels not found for ML training")
    X = df_encoded.drop(columns=['client_code','label'])
    y = df_encoded['label']
    if len(y.unique()) <= 1:
return None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    probs = model.predict_proba(X_test)
    classes = model.classes_
    top4_hits = []
    for idx, probs_row in enumerate(probs):
        label_true = y_test.iloc[idx]
        top_idx = np.argsort(probs_row)[-4:][::-1]
        top_classes = classes[top_idx]
        top4_hits.append(1 if label_true in top_classes else 0)
    precision_at_4 = float(np.mean(top4_hits))
    return model, acc, precision_at_4, (X_test, y_test)

def ml_rerank_for_client(model, client_features_df_row):
    if model is None:
        return None
    X_row = client_features_df_row.values.reshape(1, -1)
    probs = model.predict_proba(X_row)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return ranked  
def pipeline(profiles_path, tx_path, transfers_path, out_path="result_v2.csv", do_train_ml=True):
    profiles = safe_read_csv(profiles_path)
    tx = safe_read_csv(tx_path, parse_dates=['date'])
    transfers = safe_read_csv(transfers_path, parse_dates=['date'])

    required_profile_cols = {'client_code'}
    if not required_profile_cols.issubset(set(profiles.columns)):
        raise RuntimeError(f"profiles.csv must include columns {required_profile_cols}")
    df_encoded = build_feature_matrix_and_labels(profiles, tx, transfers)
    model, acc, prec4, test_split = (None, None, None, None)
    if do_train_ml:
        model, acc, prec4, test_split = train_ml_model(df_encoded)
        if model is not None:
            print(f"[ML] Accuracy (predict top1 rule label): {acc:.3f}, Precision@4 (demo): {prec4:.3f}")
        else:
            print("[ML] Not enough label variety to train model; falling back to rule-based only.")

    results = []
    encoded_columns_template = [c for c in df_encoded.columns if c not in ('client_code','label')]

    for _, p in profiles.iterrows():
        client_code = p['client_code']
        tx_c = tx[tx['client_code'] == client_code] if not tx.empty else pd.DataFrame()
        tr_c = transfers[transfers['client_code'] == client_code] if not transfers.empty else pd.DataFrame()
        top4_rule, all_benefits = rule_based_top4_and_meta(p, tx_c, tr_c)
        feats = extract_features_for_client(p, tx_c, tr_c)
        feats_row = {**feats, 'client_code': int(client_code)}
        for i in range(1,6):
            k = f'top_cat_{i}'
            if k not in feats_row:
                feats_row[k] = ""
        row_df = pd.DataFrame([feats_row])
        row_encoded = pd.get_dummies(row_df, columns=[f'top_cat_{i}' for i in range(1,6)], dummy_na=False)
        for col in encoded_columns_template:
            if col not in row_encoded.columns:
                row_encoded[col] = 0
        row_encoded = row_encoded[encoded_columns_template]
        ml_ranked = None
        if model is not None:
            ml_ranked = ml_rerank_for_client(model, row_encoded.iloc[0])
            ml_products = [pr for pr,_ in ml_ranked]
            for pr in PRODUCTS:
                if pr not in ml_products:
                    ml_products.append(pr)
            top4_ml = ml_products[:4]
            product_ml = top4_ml[0]
        else:
            top4_ml = [x[0] for x in top4_rule]
            product_ml = top4_ml[0] if top4_ml else None
        meta_map = {b[0]: b[2] for b in all_benefits}
        used_meta = meta_map.get(product_ml, {})
        push_text = generate_push_text(p, product_ml, used_meta) if product_ml else ""
        results.append({
            'client_code': int(client_code),
            'top4_rule': ";".join([x[0] for x in top4_rule]),
            'top4_ml': ";".join(top4_ml),
            'product_rule': top4_rule[0][0] if top4_rule else "",
            'product_ml': product_ml,
            'push_ml': push_text
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\n✅ Saved results to {out_path} ({len(df_res)} rows)\n")
    pd.set_option('display.max_colwidth', 220)
    print(df_res.head(8).to_string(index=False))
    return df_res, model, (acc, prec4)
if name == "main":
    if len(sys.argv) < 4:
        print("Usage: python .py profiles.csv transactions.csv transfers.csv")
        sys.exit(1)
    profiles_path = sys.argv[1]
    tx_path = sys.argv[2]
    transfers_path = sys.argv[3]
    df_out, model, metrics = pipeline(profiles_path, tx_path, transfers_path, out_path="result_v2.csv", do_train_ml=True)