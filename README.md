https://dagshub.com/dharsourav03/mlflow-tracking-server.mlflow

import dagshub
dagshub.init(repo_owner='dharsourav03', repo_name='mlflow-tracking-server', mlflow=True)

import dagshub
dagshub.init(repo_owner='dharsourav03', repo_name='mlflow-tracking-server', mlflow=True)

import dagshub
dagshub.init(repo_owner='dharsourav03', repo_name='mlflow-tracking-server', mlflow=True)


import dagshub
dagshub.init(repo_owner='dharsourav03', repo_name='mlflow-tracking-server', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)