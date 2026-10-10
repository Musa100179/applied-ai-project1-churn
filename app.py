import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Churn Risk Advisor', layout='wide')

RAW_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']

@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.joblib')
    with open('model_meta.json') as f:
        meta = json.load(f)
    return model, meta

def prepare_input(raw, columns):
    X = pd.get_dummies(raw[RAW_COLS]) # NO drop_first - SE ka main fix
    return X.reindex(columns=columns, fill_value=0).astype(float)

model, meta = load_artifacts()
COLS = meta['feature_columns']

def score(raw):
    return model.predict_proba(prepare_input(raw, COLS))[:, 1]

# Sidebar
with st.sidebar:
    st.header('Customer profile')
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    monthly = st.number_input('Monthly charges', 18.0, 120.0, 70.0)
    internet = st.selectbox('Internet service', ['Fiber optic', 'DSL', 'No'])
    services = []
    if internet!= 'No':
        services = st.multiselect('Add-on services', SERVICES, default=['OnlineSecurity'])
    payment = st.selectbox('Payment', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    phone = st.radio('Phone service', ['Yes', 'No'], horizontal=True)
    paperless = st.radio('Paperless billing', ['Yes', 'No'], horizontal=True)
    with st.expander('Demographics'):
        gender = st.radio('Gender', ['Female', 'Male'], horizontal=True)
        senior = st.checkbox('Senior citizen')
        partner = st.checkbox('Has partner')
        dependents = st.checkbox('Has dependents')
    threshold = st.slider('Contact threshold', 0.05, 0.95, float(meta.get('threshold', 0.30)), 0.05)

# Build customer
row = {'gender': gender, 'SeniorCitizen': int(senior), 'Partner': 'Yes' if partner else 'No',
       'Dependents': 'Yes' if dependents else 'No', 'tenure': tenure, 'PhoneService': phone,
       'MultipleLines': 'No', 'InternetService': internet, 'Contract': contract,
       'PaperlessBilling': paperless, 'PaymentMethod': payment,
       'MonthlyCharges': monthly, 'TotalCharges': tenure * monthly}
for s in SERVICES:
    row[s] = 'No internet service' if internet == 'No' else 'Yes' if s in services else 'No'
customer = pd.DataFrame([row])

# Main
st.title('Churn Risk Advisor v1.0')
st.caption(f"{meta['model_name']} | CV AUC {meta.get('cv_auc',0):.3f} | Decision support only")

tab1, tab2, tab3 = st.tabs(['One customer', 'Batch scoring', 'About'])
with tab1:
    p = float(score(customer)[0])
    band = 'HIGH' if p >= threshold else 'WATCH' if p >= threshold/2 else 'LOW'
    c1,c2,c3 = st.columns(3)
    c1.metric('Churn probability', f'{p:.0%}')
    c2.metric('Risk band', band)
    c3.metric('Action', 'Contact now' if band=='HIGH' else 'No action')
    st.progress(p)
    st.subheader('What would change the risk?')
    changes = [('Contract','One year'), ('Contract','Two year')]
    if payment == 'Electronic check': changes.append(('PaymentMethod','Credit card (automatic)'))
    if internet!='No' and 'TechSupport' not in services: changes.append(('TechSupport','Yes'))
    rows=[]
    for f,v in changes:
        alt = customer.copy(); alt[f]=v
        q = float(score(alt)[0])
        rows.append({'Change': f'{f}: {v}', 'New probability': round(q,3), 'Difference': round(q-p,3)})
    if rows: st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption('Associations learned from data, not guaranteed effects.')
    final = model[-1] if hasattr(model,'steps') else model
    if hasattr(final,'coef_'):
        with st.expander('Why this score? (Logistic Regression)'):
            z = model[:-1].transform(prepare_input(customer, COLS))[0] if hasattr(model,'steps') else prepare_input(customer, COLS).iloc[0]
            contrib = pd.Series(final.coef_[0]*z, index=COLS)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
            st.bar_chart(top)

with tab2:
    st.write('Upload a CSV with the original Telco columns.')
    file = st.file_uploader('Customer file', type='csv')
    if file:
        data = pd.read_csv(file)
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing: st.error(f'Missing columns: {missing}'); st.stop()
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
        data['p_churn'] = score(data)
        data['contact'] = np.where(data['p_churn']>=threshold,'Yes','')
        st.write(f"{(data['contact']=='Yes').sum()} of {len(data)} above threshold")
        st.dataframe(data.sort_values('p_churn', ascending=False).head(50))
        st.download_button('Download scored CSV', data.to_csv(index=False), 'scored.csv', 'text/csv')

with tab3:
    st.json({k:v for k,v in meta.items() if k!='feature_columns'})
    st.markdown('**Model Card:** This model predicts churn risk to support retention decisions. Trained on IBM Telco dataset.')
