import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("abalone.data.txt", header=None)
df.columns = ['Sex','Length','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight','Rings']

def rings_to_class(r):
    if r <= 7:
        return 1
    elif 8 <= r <= 10:
        return 2
    elif 11 <= r <= 15:
        return 3
    else:
        return 4

df['Class'] = df['Rings'].apply(rings_to_class)

features = ['Length','Diameter','ShellWeight','Height']
for feat in features:
    plt.figure()
    sns.boxplot(data=df, x='Class', y=feat)
    plt.title(f'{feat} distribution by Class')
    plt.xlabel('Class(1..4)')
    plt.tight_layout()
    plt.show()
    plt.close()

df['Sex_F'] = (df['Sex'] == 'F').astype(int)
df['Sex_M'] = (df['Sex'] == 'M').astype(int)

features_all = ['Sex_F','Sex_M','Length','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']
X = df[features_all].values
y = df['Class'].values - 1  

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca2 = pca.fit_transform(Xs)
plt.figure(figsize=(7,5))
scatter = plt.scatter(X_pca2[:,0], X_pca2[:,1], c=y, cmap='tab10', s=18, alpha=0.7)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Abalone PCA (2 components)')
plt.legend(*scatter.legend_elements(), title='Class')
plt.tight_layout()
plt.show()
plt.close()

num_runs = 5
max_depth_list = [3, 5, 7]
criterion_list = ['gini', 'entropy']
best_acc = -1.0
best_tree = None
best_params = None
best_split = None
results = []          
run_summaries = []    

for run in range(num_runs):
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.3, stratify=y, random_state=42+run
    )
    
   
    best_run_acc = -1.0
    best_run_params = None
    
    for depth in max_depth_list:
        for crit in criterion_list:
            clf = DecisionTreeClassifier(
                criterion=crit,
                max_depth=depth,
                min_samples_split=10,
                random_state=42+run
            )
            clf.fit(X_train, y_train)
            y_test_pred = clf.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)
            y_train_pred = clf.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)

           
            results.append({
                'run': run+1,
                'max_depth': depth,
                'criterion': crit,
                'train_acc': train_acc,
                'test_acc': test_acc
            })

            if test_acc > best_run_acc:
                best_run_acc = test_acc
                best_run_params = {
                    'run': run+1,
                    'max_depth': depth,
                    'criterion': crit,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                }

            if test_acc > best_acc:
                best_acc = test_acc
                best_tree = clf
                best_params = {'run': run+1, 'max_depth': depth, 'criterion': crit}
                best_split = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    run_summaries.append(best_run_params)

print("===== Best Decision Tree (overall) =====")
print(f"Best Test Accuracy: {best_acc:.3f}")
print("Best Parameters:", best_params)
print(export_text(best_tree, feature_names=features_all))

print("\n===== Summary of 5 Independent Runs (best per run) =====")
print("Run | max_depth | criterion | Train Acc | Test Acc")
for r in run_summaries:
    print(f"{r['run']:>3} | {r['max_depth']:>9} | {r['criterion']:^9} | {r['train_acc']:.3f}    | {r['test_acc']:.3f}")

X_temp, X_test, y_temp, y_test = train_test_split(
    Xs, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

unpruned_clf = DecisionTreeClassifier(
    criterion='entropy',
    random_state=0
)
unpruned_clf.fit(X_train, y_train)

print("\nUnpruned decision tree:")
print("Unpruned train acc:", accuracy_score(y_train, unpruned_clf.predict(X_train)))
print("Unpruned val acc:", accuracy_score(y_val, unpruned_clf.predict(X_val)))
print("Unpruned test acc:", accuracy_score(y_test, unpruned_clf.predict(X_test)))


path = DecisionTreeClassifier(
    random_state=0,
    criterion='entropy'
).cost_complexity_pruning_path(X_train, y_train)

ccp_alphas = path.ccp_alphas[:-1]

best_post = None
best_alpha = None
best_val_acc = -1.0

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(
        random_state=0,
        criterion='entropy',
        ccp_alpha=alpha
    )
    clf.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_post = clf
        best_alpha = alpha

print("\nPost-pruning best alpha:", best_alpha)
print("Post-pruning train acc:", accuracy_score(y_train, best_post.predict(X_train)))
print("Post-pruning val acc:", best_val_acc)
print("Post-pruning test acc:", accuracy_score(y_test, best_post.predict(X_test)))
print("\nBest post-pruned tree rules (ccp_alpha=%.5f):" % best_alpha)
print(export_text(best_post, feature_names=features_all))

gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)

xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=0)
xgb_clf.fit(X_train, y_train)

print("\nGradient Boosting Test Accuracy:", accuracy_score(y_test, gb.predict(X_test)))
print("XGBoost Test Accuracy:", accuracy_score(y_test, xgb_clf.predict(X_test)))

n_trees_list = [10, 50, 100, 200]
rf_results = []

print("\n===== Random Forest: Number of Trees Experiment =====")
print("Trees | Train Acc | Test Acc")

for n_trees in n_trees_list:
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=0,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_results.append((n_trees, train_acc, test_acc))
    print(f"{n_trees:5d} | {train_acc:.4f}   | {test_acc:.4f}")

best_n_trees = 100  
for var_ratio in [0.95, 0.98]:
    pca = PCA(n_components=var_ratio)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    rf_pca = RandomForestClassifier(n_estimators=best_n_trees, random_state=0, n_jobs=-1)
    rf_pca.fit(X_train_pca, y_train)
    test_acc = accuracy_score(y_test, rf_pca.predict(X_test_pca))
    print(f"RF with PCA {int(var_ratio*100)}% variance: Test Accuracy = {test_acc:.4f}")

nn_adam = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', random_state=0, max_iter=500)
nn_adam.fit(X_train, y_train)
nn_sgd = MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', random_state=0, max_iter=500)
nn_sgd.fit(X_train, y_train)

print("\nMLP Adam Test Accuracy:", accuracy_score(y_test, nn_adam.predict(X_test)))
print("MLP SGD Test Accuracy:", accuracy_score(y_test, nn_sgd.predict(X_test)))

l2_values = [0.0001, 0.001, 0.01]
for l2 in l2_values:
    nn_l2 = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', alpha=l2, random_state=0, max_iter=500)
    nn_l2.fit(X_train, y_train)
    acc = accuracy_score(y_test, nn_l2.predict(X_test))
    print(f"MLP Adam + L2(alpha={l2}) Test Accuracy: {acc:.4f}")
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv("cmc.data.txt", header=None)

df.columns = [
    'WifeAge','WifeEducation','HusbandEducation','NumChildren',
    'WifeReligion','WifeWorking','HusbandOccupation',
    'LivingIndex','MediaExposure','ContraceptiveMethod'
]

df = df.replace(['?', '$'], np.nan)
df = df.dropna()
df['ContraceptiveMethod'] = df['ContraceptiveMethod'].astype(int)

print("Dataset Loaded:", df.shape)
print(df.head())

plt.figure(figsize=(8,6))
sns.countplot(data=df, x='ContraceptiveMethod')
plt.title("Class Distribution of Contraceptive Methods")

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (CMC)")

X = df.drop('ContraceptiveMethod', axis=1)
y = df['ContraceptiveMethod'] - 1   

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    Xs, y, test_size=0.3, random_state=42, stratify=y
)

mlp_adam = MLPClassifier(
    hidden_layer_sizes=(50,),
    solver='adam',
    max_iter=500,
    random_state=0
)
mlp_adam.fit(X_train, y_train)

y_pred_adam = mlp_adam.predict(X_test)

print("\n==========Model 1: MLP Adam==========")
print("Accuracy:", accuracy_score(y_test, y_pred_adam))
print("F1 Score:", f1_score(y_test, y_pred_adam, average='weighted'))
print(classification_report(y_test, y_pred_adam))

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_adam), annot=True, cmap='Blues')
plt.title("Confusion Matrix - MLP Adam")

l2_value = 0.001   

mlp_l2 = MLPClassifier(
    hidden_layer_sizes=(50,),
    solver='adam',
    alpha=l2_value,   
    max_iter=500,
    random_state=0
)
mlp_l2.fit(X_train, y_train)
y_pred_l2 = mlp_l2.predict(X_test)
print("\n======Model 2: MLP Adam + L2 (alpha={}) (CMC) ======".format(l2_value))
print("Accuracy:", accuracy_score(y_test, y_pred_l2))
print("F1 Score:", f1_score(y_test, y_pred_l2, average='weighted'))
print(classification_report(y_test, y_pred_l2))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_l2), annot=True, cmap='Greens')
plt.title("Confusion Matrix - MLP Adam + L2 (CMC)")
plt.show()

print("\n========= Summary (Part B: CMC) =========")
print(f"MLP Adam Accuracy:       {accuracy_score(y_test, y_pred_adam):.4f}")
print(f"MLP Adam + L2 Accuracy:  {accuracy_score(y_test, y_pred_l2):.4f}")
