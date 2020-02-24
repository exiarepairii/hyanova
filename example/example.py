import hyanova

metric = 'mean_test_score'
path = './iris[GridSearchCV]Model1.csv'
df, params = hyanova.read_csv(path, metric)
importance = hyanova.analyze(df)
importance.to_csv('importance.csv')