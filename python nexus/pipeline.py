
This will start a 3-stage pipeline with a dead-letter queue (DLQ). The stages are designed to validate input, transform data, and persist results. Jobs that exhaust their retry attempts will be moved to the DLQ.
