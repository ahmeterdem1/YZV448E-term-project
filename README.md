# High Throughput PII Cleaner

This repo holds our YZV448E term project: building a **GDPR-friendly text cleaner** that can detect personally identifiable information (PII) in large batches of student essays and anonymize them before downstream use.

To run the project, simply build the docker compose setup:

```bash

docker-compose up --build

```

And then run the containers:

```bash

docker compose up -d

```

The startup of the FastAPI server may take a few minutes as it loads the ML models. You can check the logs
in the mean time via:

```bash

docker logs {container_id}

```

After you have confirmed that the startup is complete, to test API functionality,
run the below test script:

```bash
python test_api.py
```

This will test all endpoints and simulate real usage scenarios.

To get a UI to process text and files, open your browser and navigate to
`http://localhost:8000/api/v1/processor`. To see the stats dashboard, 
navigate to `http://localhost:8000/api/v1/dashboard`.

## Other endpoints

- `/api/v1/health`: Health check endpoint to verify the service is running.
- `/api/v1/docs`: Swagger UI for interactive API documentation.
- `/api/v1/process-text`: Endpoint to process raw text input for PII detection and anonymization.
- `/api/v1/queue`: Endpoint to see the state of the processing queue.
- `/api/v1/text/{task_id}`: Endpoint to get the status and result of a specific text processing task.
