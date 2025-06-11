import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

# 中文字型設定
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 讀取資料
df = pd.read_excel("processed_data (1).xlsx")

# 選擇重點欄位
features = ['檢傷級數', 'pH', '年齡', '呼吸次數', '意識程度E', '心跳', '血壓(SBP)', '血氧濃度(%)']
df = df[features + ['Y']].dropna()

# 特徵與目標變數
X = df[features]
y = df['Y']

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 SMOTE 平衡樣本數
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 建立 XGBoost 模型
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("✅ 準確率：", accuracy_score(y_test, y_pred))
print("📊 分類報告：\n", classification_report(y_test, y_pred))

# AUC 計算
y_pred_proba = model.predict_proba(X_test)
auc = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, multi_class='ovr')
print("🔥 AUC：", auc)

# SHAP 特徵解釋
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, features=features)