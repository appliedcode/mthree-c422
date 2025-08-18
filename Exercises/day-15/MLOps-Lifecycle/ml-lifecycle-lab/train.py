import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
           'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
df = pd.read_csv(url, header=None, names=columns, na_values=' ?')
df = df.dropna()
df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model and column list
dump(model, 'model.joblib')
dump(list(X_train.columns), 'columns.joblib')
