import pandas as pd
from  sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score



#df = pd.read_csv(path)
df = pd.read_csv('pysstubs.csv')
print(df.shape)
print(df.columns.tolist())
print(df.head(3))

total_rows = df['sstub_pattern'].size

print("\n")

print(total_rows)

print(df["line_before"].isna().sum())


duplicate_rows_count = len(df) - df["commit"].nunique()
print(f"Number of rows sharing a commit value: {duplicate_rows_count}\n")

print(df["sstub_pattern"].value_counts())
df = df.dropna(subset='line_before')

groups=df["project_name"]
X = df["line_before"]
y = df["sstub_pattern"]


gss = GroupShuffleSplit(
    n_splits=1,
   test_size=0.2,
    random_state=42,
)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx] , y.iloc[test_idx]

group_overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
print(f"\n Overlap groups: {group_overlap}")


dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
pred = dummy.predict(X_test)

print(accuracy_score(y_test, pred))
print(f1_score(y_test, pred, average="macro"))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50000)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)


print(X_train_vec.shape)

model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train_vec, y_train)

pred = model1.predict(X_test_vec)

print(accuracy_score(y_test, pred))
print(f1_score(y_test, pred, average="macro"))

from sklearn.metrics import classification_report
print(classification_report(y_test, pred, zero_division=0))

