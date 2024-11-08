from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
    # Dict for mapping the months and weekdays
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Drop na, create new columns for onehot encode and drop the original 'job' column
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    # Change the string into binary form is 1 == yes, 0 == no for columns
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    # Drop na, create new columns for onehot encode and drop the original 'contact' column
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    # Drop na, create new columns for onehot encode and drop the original 'education' column
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    # Replace strings with number as map define above
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    # Change the string into binary form is 1 == yes, 0 == no for columns
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    # Taken steps seem to be independence from each other, but arrange inefficient
    # But wanna make sure the code work and have limit of time this account so not gonna change a thing
    # Just comments for future reference
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse regularization strength, smaller values mean stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    # TODO: Create TabularDataset using TabularDatasetFactory

    url = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    ds = TabularDatasetFactory.from_delimited_files(path=url)
    
    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    root_mean_squared_error = model.score(x_test, y_test)
    run.log("root_mean_squared_error", np.float(root_mean_squared_error))

    # Save the model to the outputs folder for deployment
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model_'+'accuracy_'+str(root_mean_squared_error)+"_"+'C_'+str(args.C)+"_"+'maxIter_'+str(args.max_iter)+'.joblib')

if __name__ == '__main__':
    main()