FROM apache/spark:3.5.0

USER root
WORKDIR /app

# Install ML libraries sequentially to leverage cache and avoid timeout
RUN pip install --upgrade pip --default-timeout=1000
RUN pip install numpy --default-timeout=1000
RUN pip install pandas --default-timeout=1000
RUN pip install scikit-learn --default-timeout=1000
RUN pip install joblib --default-timeout=1000
RUN pip install ta --default-timeout=1000
RUN pip install multitasking==0.0.9 --default-timeout=1000
RUN pip install yfinance==0.2.40 --default-timeout=1000

COPY processor.py .
COPY train_model.py .

# No extra pip requirements needed for basic pyspark execution in this image
# but if needed later, we can add them here.
