# Use AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Copy requirements and install into Lambda task root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy your function code and model into Lambda task root
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY model.pkl ${LAMBDA_TASK_ROOT}

# Command to run the Lambda handler
CMD ["lambda_function.lambda_handler"]
