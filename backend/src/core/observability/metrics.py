"""
Observability Metrics - Community Edition Stub

The Community Edition does not include Prometheus metrics.
Enterprise Edition includes full observability with Prometheus, Grafana, and Jaeger.
"""

class _DummyMetric:
    """Dummy metric that does nothing"""
    def inc(self, *args, **kwargs):
        pass

    def dec(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def labels(self, *args, **kwargs):
        return self

# Dummy metrics for Community Edition
query_total = _DummyMetric()
query_latency = _DummyMetric()
query_context_chunks = _DummyMetric()
reranker_enabled = _DummyMetric()
llm_tokens_total = _DummyMetric()
llm_cost_usd = _DummyMetric()
llm_latency = _DummyMetric()
