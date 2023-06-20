from utils import get_df_from_db, get_engine
from ydata_profiling import ProfileReport
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import joblib
import dotenv

def format_df(df):
    X_cols = [
    'Have_Stream_Energy',
    'Have_NaturalGas_Energy', 
    'PrimaryPropertyType',
    'NumberofBuildings',
    'LargestPropertyUseTypeGFA_log',
    ]
    
    X = df[X_cols]
    y_multi = df[['TotalGHGEmissions_log', 'SiteEnergyUse_kBtu_log']]
    y1 = df['TotalGHGEmissions_log']
    return X, y_multi, y1

def train_grid(grid, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid.fit(X_train, y_train)
    return grid.best_estimator_
    

def make_pipe(X_cat, X_num):
    nn = MLPRegressor(early_stopping=True)
    preparation = ColumnTransformer(transformers=[
    ('tf_cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), X_cat.columns),
    ('tf_num', StandardScaler(), X_num.columns)
    ])
    pipe_nn= Pipeline(steps=[
    ('preparation', preparation),
    ('model', nn)
    ])
    params_nn = {'model__solver': ['adam', 'sgd'], 
             'model__hidden_layer_sizes': [60]}
    grid = GridSearchCV(pipe_nn, params_nn, scoring='r2', n_jobs=-1, verbose=2)
    return grid

def multi_output_reg(estimator, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultiOutputRegressor(estimator)
    model.fit(X_train, y_train)
    return model

def main():
    dotenv.load_dotenv()
    # create report.html
    engine = get_engine(echo_arg=True)

    df = get_df_from_db(engine)

    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("templates/report.html")
    print('report.html has been generated')
    
    # create model.pkl
    X, y_multi, y1 = format_df(df)
    X_cat = X.select_dtypes(include=[object, bool])
    X_num = X.select_dtypes(exclude=[object, bool])
    
    grid = make_pipe(X_cat, X_num)
    estimator = train_grid(grid, X, y1)
    model = multi_output_reg(estimator, X, y_multi)
    joblib.dump(model, 'model.pkl')
    print('model has been saved')
if __name__=='__main__':
    main()