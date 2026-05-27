FROM python:3.11
WORKDIR /home/app/
COPY ./ /home/app/
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r ./requirements.txt --no-deps
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

#ENV LM_STUDIO_HOST=localhost
#ENV LM_STUDIO_PORT=1234