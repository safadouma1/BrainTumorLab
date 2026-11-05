FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -e .
ENV BTL_BASE=/data BTL_DATA_ROOT=/data/BraTS_2020_Data_preprocessing_nii_FIXED
EXPOSE 8000
CMD ["python", "-m", "braintumorlab.webapp.server", "--port", "8000"]
