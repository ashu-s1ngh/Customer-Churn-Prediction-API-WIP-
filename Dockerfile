# Stage 1
FROM amazonlinux:2023 AS builder
WORKDIR /build
RUN dnf update -y && \
    dnf install -y gcc-c++ python3.11-devel python3.11-pip && \
    dnf clean all

COPY requirements.txt .
RUN pip3.11 install --no-cache-dir --target="./packages" -r requirements.txt

# Stage 2
FROM public.ecr.aws/lambda/python:3.11
WORKDIR /var/task
COPY --from=builder /build/packages/ .
COPY main.py .
COPY preprocessor_pipeline.joblib .
COPY churn_model.joblib .
CMD ["main.handler"]