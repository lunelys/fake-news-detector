# Monitoring (Grafana + Prometheus)

1. Start the FastAPI service on port 8000.
2. From this folder, run:
   `docker compose up -d`
3. Open Grafana on http://localhost:3000 (admin / admin).
4. Add Prometheus datasource at http://prometheus:9090.

If you run the API on a different host or port, update `prometheus.yml`.
