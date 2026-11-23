FROM public.ecr.aws/lambda/python:3.12
# libgomp is required by LightGBM
RUN dnf install -y libgomp && dnf clean all
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt
COPY lambda_function.py sweep.py polymarket.py odds_history.py lib.py build_segments.py build_cand2.py model.py team_translations.py \
     model_full.pkl big_leagues.json service-account.json ${LAMBDA_TASK_ROOT}/
COPY winner_slack_model/ ${LAMBDA_TASK_ROOT}/winner_slack_model/
COPY oddsflow/ ${LAMBDA_TASK_ROOT}/oddsflow/
CMD [ "lambda_function.lambda_handler" ]
