# ============ Week 4 | Cell 22: Save meta + sample ============
meta = {
    'model_name': type(final).__name__,
    'sklearn_version': sklearn.__version__,
    'feature_columns': cols,
    'threshold': 0.30,
    'cv_auc': 0.845,         # 👈 apna Week 3 CV number
    'cv_auc_std': 0.012,     # 👈 apna
    'test_auc': 0.840,       # 👈 apna
    'training_data': 'IBM Telco Customer Churn, 7,043 customers',
    'version': '1.0',
}

with open('/kaggle/working/model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

# Save 50 sample customers for batch demo
sample = df.drop(columns=['Churn']).sample(50, random_state=1)
sample.to_csv('/kaggle/working/sample_customers.csv', index=False)

print('✅ Saved model_meta.json')
print('✅ Saved sample_customers.csv')
print('\n📁 Files in /kaggle/working:')
import os
for f in os.listdir('/kaggle/working'):
    print('  -', f)
