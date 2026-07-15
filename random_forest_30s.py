import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import io
import contextlib
import warnings

warnings.filterwarnings('ignore')


# 1. CARICAMENTO E PULIZIA AVANZATA DEI DATI
nome_file_csv = 'dati_estratti_TUTTI.csv'

print(f"Caricamento del file '{nome_file_csv}'...")
df = pd.read_csv(nome_file_csv, sep=',', decimal='.')

# Rimozione colonne vuote e rimozione spazi vuoti nei nomi delle colonne
df = df.dropna(how='all', axis=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

# Tuttle le colonne sono forzate a essere numeriche
colonne_numeriche = ['ora', 'minuti', 'secondi', 'millisecondi', 'mvmt', 'motion_index', 'picco', 'rssi', 'pkt_s', 'headcount']
for col in colonne_numeriche:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Rimozione record nulli
df = df.dropna(subset=['mvmt', 'motion_index', 'headcount']).reset_index(drop=True)
df['headcount'] = df['headcount'].astype(int)
df['picco'] = df['picco'].astype(int)

# Ordinamento cronologico 
df = df.sort_values(by=['ora', 'minuti', 'secondi', 'millisecondi']).reset_index(drop=True)
print(f"Dataset caricato con successo: {len(df)} righe rilevate.")


# 2. SEGMENTAZIONE DELLE FINESTRE TEMPORALI 

timestamp_in_s = df['ora'] * 3600 + df['minuti'] * 60 + df['secondi'] + df['millisecondi'] / 1000

session_id = 0
sessions = []
last_t = timestamp_in_s.iloc[0]

# Rilevamento automatico delle sessioni di test
for t in timestamp_in_s:
    if abs(t - last_t) > 300: 
        session_id += 1
    sessions.append(session_id)
    last_t = t

df['session_id'] = sessions
df['timestamp_s'] = timestamp_in_s

# Dimensione della finestra scelta per il confronto
DURATA_FINESTRA = 30

df['window_id'] = df.groupby('session_id').apply(
    lambda x: (x['timestamp_s'] - x['timestamp_s'].min()) // DURATA_FINESTRA
).reset_index(level=0, drop=True).astype(int)

df['unique_window_id'] = df['session_id'].astype(str) + "_" + df['window_id'].astype(str)


# 3. ESTRAZIONE DELLE FEATURE AGGREGATE PER FINESTRA 
def calcola_features_finestra(group):
    if len(group) < 3: 
        return None
    
    diz_feat = {}
    
    # Feature statistiche sulla colonna MVMT 
    diz_feat['mvmt_mean'] = group['mvmt'].mean()
    diz_feat['mvmt_std'] = group['mvmt'].std(ddof=0)
    diz_feat['mvmt_min'] = group['mvmt'].min()
    diz_feat['mvmt_max'] = group['mvmt'].max()
    diz_feat['mvmt_range'] = diz_feat['mvmt_max'] - diz_feat['mvmt_min']
    diz_feat['mvmt_p90'] = group['mvmt'].quantile(0.9)
    
    # Feature statistiche sulla colonna MOTION INDEX 
    diz_feat['motion_mean'] = group['motion_index'].mean()
    diz_feat['motion_std'] = group['motion_index'].std(ddof=0)
    diz_feat['motion_min'] = group['motion_index'].min()
    diz_feat['motion_max'] = group['motion_index'].max()
    diz_feat['motion_range'] = diz_feat['motion_max'] - diz_feat['motion_min']
    diz_feat['motion_p90'] = group['motion_index'].quantile(0.9)
    diz_feat['motion_energy'] = np.sum(group['motion_index'] ** 2)
    
    # Altri parametri
    diz_feat['rssi_mean'] = group['rssi'].mean()
    diz_feat['pkt_s_mean'] = group['pkt_s'].mean()
    diz_feat['num_picchi'] = group['picco'].sum()
    
    if 'state' in group.columns: #stato prevalente
        stato_numerico = group['state'].astype(str).str.strip().map({'IDLE': 0, 'MOTION': 1}).fillna(0)
        diz_feat['state_prevalente'] = 1 if stato_numerico.mean() >= 0.5 else 0
    else:
        diz_feat['state_prevalente'] = 0
    
    # TARGET: headcount 
    diz_feat['headcount'] = int(group['headcount'].mode()[0])
    
    return pd.Series(diz_feat)

print(f"Aggregazione dati e calcolo feature su finestre stabili da {DURATA_FINESTRA} secondi...")
df_features = df.groupby('unique_window_id').apply(calcola_features_finestra).dropna().reset_index(drop=True)


# 4. PREPARAZIONE DEI DATI PER L'ADDESTRAMENTO

X = df_features.drop(columns=['headcount'])
y = df_features['headcount'].astype(int)

# Controllo di stabilità sulle classi
classi_valide = y.value_counts()
classi_insufficienti = classi_valide[classi_valide < 2].index.tolist()
if classi_insufficienti:
    maschera_filtraggio = ~y.isin(classi_insufficienti)
    X = X[maschera_filtraggio]
    y = y[maschera_filtraggio]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. ADDESTRAMENTO E MIGLIORAMENTO IPERPARAMETRI (GRID SEARCH)

print("\nEsecuzione Grid Search per RANDOM FOREST...")

# Inserito class_weight='balanced' per supportare al meglio la classe 2
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)

# Spazio iperparametri esteso per una ottimizzazione robusta
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid_rf = GridSearchCV(estimator=rf_base, param_grid=param_grid_rf, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

# Estrazione del modello ottimo post-ottimizzazione
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)


# 6. GENERAZIONE DEL REPORT

output_buffer = io.StringIO()
with contextlib.redirect_stdout(output_buffer):
    print("=" * 85)
    print(f"      REPORT: RANDOM FOREST CLASSIFIER - FINESTRE {DURATA_FINESTRA}s")
    print("=" * 85)
    print(f"Dataset originale: {len(df)} record istantanei (righe grezze).")
    print(f"Dataset aggregato generato: {len(X)} finestre temporali totali.")
    
    print("\n[1] DISTRIBUZIONE DELLE FINESTRE TEMPORALI PER CLASSE TARGET:")
    for cls, count in y.value_counts().sort_index().items():
        print(f"  - Classe {cls} (Persone presenti): {count} finestre ({count/len(X)*100:.2f}%)")
        
    print("\n[2] RISULTATI DELL'OTTIMIZZAZIONE IPERPARAMETRI (GRID SEARCH):")
    print(f"  - Migliore configurazione di parametri trovata: {grid_rf.best_params_}")
    print(f"  - Accuratezza media stimata in Cross-Validation : {grid_rf.best_score_:.4f}")
    
    print("\n[3] PERFORMANCE SUL TEST SET GENERALE (DATI MAI VISTI):")
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"  - Accuracy Score complessiva di Random Forest: {acc_rf:.4f} ({acc_rf*100:.2f}%)")
    
    print("\n[4] METRICHE DI CLASSIFICAZIONE ANALITICHE (PRECISION, RECALL, F1):")
    print(classification_report(y_test, y_pred_rf))
    
    print("[5] MATRICE DI CONFUSIONE (CONTEGGIO DELLE FINESTRE REALMENTE ASSEGNATE):")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    classes_labels = sorted(y.unique())
    header_pred = " ".join([f"Predetto {c}".center(14) for c in classes_labels])
    print(f"                {header_pred}")
    for idx, row in enumerate(cm_rf):
        row_str = " ".join([f"{val}".center(14) for val in row])
        print(f"  Reale {classes_labels[idx]} pers. :    {row_str}")
    
    print("\n[6] GERARCHIA DI IMPORTANZA DELLE FEATURE (FEATURE IMPORTANCE STRUTTURALE):")
    importances_rf = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    for index, (feat, imp) in enumerate(importances_rf.items(), start=1):
        print(f"  {index:02d}. {feat:<22}: {imp:.4f} ({imp*100:.2f}%)")
    print("=" * 85)

# Scrittura del file
nome_file_report = 'report_dettagliato_rf.txt'
with open(nome_file_report, 'w', encoding='utf-8') as f:
    f.write(output_buffer.getvalue())

print(f"\n[COMPLETATO] Nuovo report per Random Forest generato con successo.")
print(f"Il file di testo è pronto in: '{nome_file_report}'")