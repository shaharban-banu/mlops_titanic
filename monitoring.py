import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_drift_check():
    
    ref=pd.read_csv("data/titanic.csv").drop(columns=["Survived"])
    currr=pd.read_csv("data/current.csv")
    print("REF COLUMNS:", ref.columns)
    print("CURR COLUMNS:", currr.columns)
    report=Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref,current_data=currr)
    report.save_html("drift_report.html")

if __name__=="__main__":
    run_drift_check()
