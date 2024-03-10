import pandas as pd
import numpy as np
import mlflow
import pickle
from datetime import datetime
import sys
import os
import shutil

## NOTE: Uncomment to use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.
# mlflow.set_tracking_uri("http://training.itu.dk:5000")

# # Setting the requried environment variables
# os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://130.226.140.28:5000'
# os.environ['AWS_ACCESS_KEY_ID'] = 'training-bucket-access-key'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'tqvdSsEDnBWTDuGkZYVsRKnTeu'

# Set the experiment name
experiment_name = "krpa_a2 - logging the model"
mlflow.set_experiment("best_model_saving")

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Start a run
def start_run(run_name, poly, regressor, run_id=None):
    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        # Insert path to dataset
        df = pd.read_json("dataset.json", orient="split")

        # Handle missing data
        total = df["Total"]
        total = total.resample('3H').mean()
        cleaned = df[~df.isnull().any(axis=1)]
        merged = pd.merge(cleaned, total, left_index=True, right_index=True)
        merged.rename(columns={'Total_y':'Power'}, inplace=True)
        merged.drop(columns={'Total_x'}, inplace=True)
        df = merged

        #helper functions
        def dir_to_trig(X):

            """Converts wind direction to sin. Makes sure directions such as N and WNW are mapped closed to each other"""

            degrees_map = {
                'N': 0,
                'NNE': 22.5,
                'NE': 45,
                'ENE': 67.5,
                'E': 90,
                'ESE': 112.5,
                'SE': 135,
                'SSE': 157.5,
                'S': 180,
                'SSW': 202.5,
                'SW': 225,
                'WSW': 247.5,
                'W': 270,
                'WNW': 292.5,
                'NW': 315,
                'NNW': 337.5,
            }

            #map directions to degrees
            X['Direction'] = X['Direction'].map(degrees_map)

            #convert degrees to radians, then to sin
            X['Direction_sin'] = np.sin(X['Direction'] * np.pi / 180)

            return X[['Direction_sin']]
        
        preprocessing = ColumnTransformer([
            #non-specified columns are dropped by default
            ('dir_to_trig', FunctionTransformer(dir_to_trig), ['Direction']),
            ('speed_scaler', StandardScaler(), ['Speed'])
            ])
        
        pipeline = Pipeline([
            # Piece of the original pipeline
            ('preprocessing', preprocessing),
            ('poly', PolynomialFeatures(poly)),
            ('regressor', regressor)
        ])

        metrics = [
            ("MAE", mean_absolute_error, []),
            ('MSE', mean_squared_error, []),
            ("R2", r2_score, [])
        ]

        X = df[["Speed","Direction"]]
        y = df["Power"]

        number_of_splits = 5

        #Log your parameters. What parameters are important to log?
        mlflow.log_param("polynomial_degree", poly)
        mlflow.log_param("regressor", pipeline.steps[2][1])

        model = pipeline.fit(X,y)

        #Only for the final one
        if run_id != None:
            mlflow.sklearn.log_model(model, "best_model")
            print(f"run_id: {run_id}")
            experiment_id="0" #presuming the default
            print(f"experiment_id: {experiment_id}")
            print(f"path to model: ./mlruns/{experiment_id}/{run_id}/artifacts/best_model")
        
        #Evaluation, this part doesn't produce a model.
        for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
            pipeline.fit(X.iloc[train],y.iloc[train])
            predictions = pipeline.predict(X.iloc[test])
            truth = y.iloc[test]
            
            # Calculate and save the metrics for this fold
            for name, func, scores in metrics:
                score = func(truth, predictions)
                scores.append(score)
        
        # Log a summary of the metrics
        if run_id == None:
            print("\n", run_name)
            for name, _, scores in metrics:
                # NOTE: Here we just log the mean of the scores. 
                # Are there other summarizations that could be interesting?
                # Median - explain that in the report, but this doesn't really matter here, mean is still the SOTA go-to with cross-validation.
                mean_score = sum(scores)/number_of_splits
                print(f"mean{name}", mean_score)
                mlflow.log_metric(f"mean_{name}", mean_score)


# Automatic Parameter Generation
def main():
    for poly in range (1,6):
        for regressor in (LinearRegression(), RandomForestRegressor(random_state=30), SVR()):
            start_run(f"{poly}_{regressor}", poly, regressor) 

    #find and save the best model
    df = mlflow.search_runs()
    run_id, poly, regressor = df.loc[df['metrics.mean_R2'].idxmax()][['run_id', 'params.polynomial_degree', 'params.regressor']]
    start_run("", int(poly), eval(regressor), run_id) #eval() is safe for use here as it's not user-induced

if __name__=="__main__":
     main()


