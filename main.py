import pandas as pd
import xgboost as xgb
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# 中文字型設定
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 讀取資料
df = pd.read_excel("processed_data (1).xlsx")

# ===== 模型分析 =====
features = ['檢傷級數', 'pH', '年齡', '呼吸次數', '意識程度E', '心跳', '血壓(SBP)', '血氧濃度(%)']
df_model = df[features + ['Y']].dropna()
X = df_model[features]
y = df_model['Y']

# 標準化與平衡
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_scaled, y)

# 訓練與預測
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
model = xgb.XGBClassifier(
    objective='multi:softprob', num_class=3,
    eval_metric='mlogloss', use_label_encoder=False, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ 準確率：", accuracy_score(y_test, y_pred))
print("📊 分類報告：\n", classification_report(y_test, y_pred))
print("🔥 AUC：", roc_auc_score(pd.get_dummies(y_test), model.predict_proba(X_test), multi_class='ovr'))

# ===== SHAP 圖表 =====
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

plt.figure()
shap.summary_plot(shap_values, features=features, show=False)
plt.title("SHAP 特徵重要性圖")
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()
print("✅ SHAP 圖已儲存為 shap_summary.png")

# 患者基本資料分布圖（連續變數）
basic_features = ['年齡', '心跳', '呼吸次數', '血壓(SBP)', '血氧濃度(%)', 'pH']
df[basic_features].hist(bins=20, figsize=(12, 8), grid=False)
plt.suptitle("患者基本資料分布", fontsize=16)
plt.tight_layout()
plt.savefig("basic_stats.png")
plt.close()

# ===== 統計圖表儲存 =====
df_plot = df[['檢傷級數', '年齡', '意識程度E', 'Y']].dropna()

# 1️⃣ 檢傷級數 vs 住院等級
plt.figure(figsize=(6, 4))
sns.countplot(data=df_plot, x='檢傷級數', hue='Y', palette='Set2')
plt.title('檢傷級數與住院等級分布')
plt.xlabel('檢傷級數')
plt.ylabel('人數')
plt.tight_layout()
plt.savefig("triage_vs_admission.png")
plt.close()
print("📊 已儲存：triage_vs_admission.png")

# 2️⃣ 意識程度 vs 住院等級
plt.figure(figsize=(6, 4))
sns.countplot(data=df_plot, x='意識程度E', hue='Y', palette='Set2')
plt.title('意識程度與住院等級分布')
plt.xlabel('意識程度 (E)')
plt.ylabel('人數')
plt.tight_layout()
plt.savefig("consciousness_vs_admission.png")
plt.close()
print("📊 已儲存：consciousness_vs_admission.png")

# 3️⃣ 年齡分組 vs 住院等級
df_plot['年齡分組'] = pd.cut(df_plot['年齡'], bins=[0, 30, 60, 90, 120], right=False)
plt.figure(figsize=(6, 4))
sns.countplot(data=df_plot, x='年齡分組', hue='Y', palette='Set2')
plt.title('年齡分組與住院等級分布')
plt.xlabel('年齡分組')
plt.ylabel('人數')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("agegroup_vs_admission.png")
plt.close()
print("📊 已儲存：agegroup_vs_admission.png")

print("✅ 所有圖表已產生並儲存在資料夾中。")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===== 混淆矩陣圖 =====
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

plt.figure(figsize=(6, 5))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("住院狀態預測混淆矩陣")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("📊 混淆矩陣圖已儲存為 confusion_matrix.png")
