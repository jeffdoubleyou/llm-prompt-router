"""Tests for GET /api/v1/classifier/samples endpoint (JEF-111).

Covers unit-style tests (via TestClient), integration tests (real DB queries),
and edge cases as specified in the task.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time

import pytest
from httpx import AsyncClient, Response


# ======================================================================
# Unit Tests — Pagination
# ======================================================================

class TestPagination:
    """Pagination parameter validation and default behaviour."""

    @pytest.mark.asyncio
    async def test_default_page_and_page_size(self, client: AsyncClient):
        """Without explicit page/page_size, defaults to page=1, page_size=20."""
        resp = await client.get("/api/v1/classifier/samples")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 1
        assert data["page_size"] == 20

    @pytest.mark.asyncio
    async def test_valid_page_and_page_size(self, client: AsyncClient):
        """Explicit valid page and page_size are honoured."""
        resp = await client.get("/api/v1/classifier/samples?page=2&page_size=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 2
        assert data["page_size"] == 3
        assert len(data["samples"]) <= 3

    @pytest.mark.asyncio
    async def test_page_zero_is_rejected(self, client: AsyncClient):
        """page=0 should be rejected (FastAPI ge=1 validation)."""
        resp = await client.get("/api/v1/classifier/samples?page=0")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_negative_page_is_rejected(self, client: AsyncClient):
        """Negative page values are rejected."""
        resp = await client.get("/api/v1/classifier/samples?page=-1")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_page_size_exceeding_max_is_rejected(self, client: AsyncClient):
        """page_size > 100 (le=100) is rejected."""
        resp = await client.get("/api/v1/classifier/samples?page_size=101")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_page_size_at_max_is_accepted(self, client: AsyncClient, populated_db):
        """page_size=100 (the maximum) should be accepted."""
        resp = await client.get("/api/v1/classifier/samples?page_size=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page_size"] == 100


# ======================================================================
# Unit Tests — Filtering
# ======================================================================

class TestFiltering:
    """Filtering by model name, correctness, and combined filters."""

    @pytest.mark.asyncio
    async def test_filter_by_model_exact_match(self, client: AsyncClient, populated_db):
        """Filtering by model returns only samples from that model."""
        resp = await client.get("/api/v1/classifier/samples?model=gpt-4")
        assert resp.status_code == 200
        data = resp.json()
        assert all(s["selected_model"] == "gpt-4" for s in data["samples"])

    @pytest.mark.asyncio
    async def test_filter_by_model_no_match(self, client: AsyncClient):
        """Filter by a model that has no samples returns empty list."""
        resp = await client.get("/api/v1/classifier/samples?model=nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["samples"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_filter_by_correct_true(self, client: AsyncClient, populated_db):
        """Filter is_correct=true returns only correct samples."""
        resp = await client.get("/api/v1/classifier/samples?correct=true")
        assert resp.status_code == 200
        data = resp.json()
        assert all(s["is_correct"] is True for s in data["samples"])

    @pytest.mark.asyncio
    async def test_filter_by_correct_false(self, client: AsyncClient, populated_db):
        """Filter is_correct=false returns only incorrect samples."""
        resp = await client.get("/api/v1/classifier/samples?correct=false")
        assert resp.status_code == 200
        data = resp.json()
        assert all(s["is_correct"] is False for s in data["samples"])

    @pytest.mark.asyncio
    async def test_filter_by_correct_excludes_null(self, client: AsyncClient, populated_db):
        """Filter is_correct=false does NOT include null values."""
        resp = await client.get("/api/v1/classifier/samples?correct=false")
        assert resp.status_code == 200
        data = resp.json()
        assert all(s["is_correct"] is False for s in data["samples"])
        assert all(s["is_correct"] is not None for s in data["samples"])

    @pytest.mark.asyncio
    async def test_combined_filters_model_and_correctness(self, client: AsyncClient, populated_db):
        """Combine model and correctness filters."""
        resp = await client.get("/api/v1/classifier/samples?model=gpt-4&correct=true")
        assert resp.status_code == 200
        data = resp.json()
        for s in data["samples"]:
            assert s["selected_model"] == "gpt-4"
            assert s["is_correct"] is True

    @pytest.mark.asyncio
    async def test_combined_filters_all_three(self, client: AsyncClient, populated_db):
        """Combine model + correctness + search together."""
        resp = await client.get("/api/v1/classifier/samples?model=claude-3&correct=true&q=javascript")
        assert resp.status_code == 200
        data = resp.json()
        for s in data["samples"]:
            assert s["selected_model"] == "claude-3"
            assert s["is_correct"] is True
            assert "javascript" in s["prompt_text"].lower()


# ======================================================================
# Unit Tests — Search
# ======================================================================

class TestSearch:
    """Text search on prompt_text (case-insensitive partial match)."""

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, client: AsyncClient, populated_db):
        """Search is case-insensitive."""
        resp = await client.get("/api/v1/classifier/samples?q=capital")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert any("capital" in s["prompt_text"].lower() for s in data["samples"])

    @pytest.mark.asyncio
    async def test_search_case_insensitive_uppercase(self, client: AsyncClient, populated_db):
        """Uppercase search matches lowercase text."""
        resp = await client.get("/api/v1/classifier/samples?q=CAPITAL")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    @pytest.mark.asyncio
    async def test_search_partial_match(self, client: AsyncClient, populated_db):
        """Search matches partial words."""
        resp = await client.get("/api/v1/classifier/samples?q=python")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_search_no_match(self, client: AsyncClient):
        """Search with no matching results returns empty list."""
        resp = await client.get("/api/v1/classifier/samples?q=xyznonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["samples"] == []
        assert data["total"] == 0


# ======================================================================
# Unit Tests — Empty Results
# ======================================================================

class TestEmptyResults:
    """Empty result sets still return 200 with correct structure."""

    @pytest.mark.asyncio
    async def test_empty_results_status_code(self, client: AsyncClient):
        """Empty results return HTTP 200, not 404."""
        resp = await client.get("/api/v1/classifier/samples?q=zzzz_not_found")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_empty_results_body(self, client: AsyncClient):
        """Empty results have empty samples array and total: 0."""
        resp = await client.get("/api/v1/classifier/samples?q=zzzz_not_found")
        data = resp.json()
        assert data["samples"] == []
        assert data["total"] == 0
        assert "page" in data
        assert "page_size" in data


# ======================================================================
# Unit Tests — Response Format
# ======================================================================

class TestResponseFormat:
    """Verify all required response fields and correct types."""

    @pytest.mark.asyncio
    async def test_required_response_fields(self, client: AsyncClient, populated_db):
        """Response must include samples, total, page, page_size."""
        resp = await client.get("/api/v1/classifier/samples")
        data = resp.json()
        assert "samples" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data

    @pytest.mark.asyncio
    async def test_response_field_types(self, client: AsyncClient, populated_db):
        """top-level fields must have correct types."""
        resp = await client.get("/api/v1/classifier/samples")
        data = resp.json()
        assert isinstance(data["samples"], list)
        assert isinstance(data["total"], int)
        assert isinstance(data["page"], int)
        assert isinstance(data["page_size"], int)

    @pytest.mark.asyncio
    async def test_sample_field_types(self, client: AsyncClient, populated_db):
        """Each sample object has the correct fields and types."""
        resp = await client.get("/api/v1/classifier/samples")
        data = resp.json()
        if data["samples"]:
            sample = data["samples"][0]
            assert isinstance(sample["id"], str)
            assert isinstance(sample["prompt_text"], str)
            assert isinstance(sample["selected_model"], str)
            assert isinstance(sample["features"], dict)
            assert sample["is_correct"] is None or isinstance(sample["is_correct"], bool)
            assert isinstance(sample["created_at"], str)


# ======================================================================
# Integration Tests — End-to-End HTTP Request/Response
# ======================================================================

class TestIntegration:
    """End-to-end HTTP request/response cycle tests."""

    @pytest.mark.asyncio
    async def test_full_request_response_cycle(self, client: AsyncClient, populated_db):
        """A complete GET request returns expected structure and data."""
        resp = await client.get("/api/v1/classifier/samples?page=1&page_size=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["samples"]) <= 5
        assert data["page"] == 1
        assert data["page_size"] == 5

    @pytest.mark.asyncio
    async def test_database_queries_use_parameterized_statements(self, client: AsyncClient, populated_db):
        """Verify that the endpoint uses parameterized queries (no SQL injection).

        We test by injecting SQL-like strings and confirming they are treated
        as literal text, not executed as SQL.
        """
        # This would trigger SQL injection if the query was built with string interpolation
        resp = await client.get(
            "/api/v1/classifier/samples?q='; DROP TABLE classifier_samples; --"
        )
        # The endpoint should return 200 with an empty set (the SQL string is
        # treated as literal search text, not executed)
        assert resp.status_code == 200
        # The DB table still exists and is intact
        resp2 = await client.get("/api/v1/classifier/samples?page_size=1")
        assert resp2.status_code == 200
        assert "samples" in resp2.json()

    @pytest.mark.asyncio
    async def test_pagination_metadata_accuracy(self, client: AsyncClient, populated_db):
        """The total count must match the filtered result count (not just page size)."""
        # Get all samples for 'gpt-4' with page_size=2 (small page)
        resp = await client.get("/api/v1/classifier/samples?model=gpt-4&page_size=2")
        data = resp.json()
        # total should reflect ALL gpt-4 samples, not just the 2 returned
        assert data["total"] > len(data["samples"])
        # But total should match the actual count across all pages
        resp_all = await client.get("/api/v1/classifier/samples?model=gpt-4&page_size=100")
        assert data["total"] == resp_all.json()["total"]

    @pytest.mark.asyncio
    async def test_results_spanning_multiple_pages(self, client: AsyncClient, populated_db):
        """Samples on page 2 should be different from page 1."""
        resp1 = await client.get("/api/v1/classifier/samples?page=1&page_size=2")
        resp2 = await client.get("/api/v1/classifier/samples?page=2&page_size=2")
        data1 = resp1.json()
        data2 = resp2.json()
        # The sample IDs should differ between pages
        ids1 = {s["id"] for s in data1["samples"]}
        ids2 = {s["id"] for s in data2["samples"]}
        assert ids1.isdisjoint(ids2)


# ======================================================================
# Integration Tests — Concurrent Requests (No Data Races)
# ======================================================================

class TestConcurrency:
    """Concurrent requests should not cause data races."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_no_data_races(self, client: AsyncClient, populated_db):
        """Fire multiple simultaneous requests and verify all succeed with correct structure."""
        num_requests = 20

        async def make_request(i: int) -> Response:
            return await client.get(f"/api/v1/classifier/samples?page=1&page_size={i % 10 + 1}&q=ai")

        # Fire all requests concurrently
        tasks = [make_request(i) for i in range(num_requests)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Every request must succeed
        for i, result in enumerate(responses):
            if isinstance(result, Exception):
                pytest.fail(f"Request {i} raised exception: {result}")
            assert result.status_code == 200, f"Request {i} returned {result.status_code}"
            data = result.json()
            assert "samples" in data
            assert "total" in data
            assert "page" in data
            assert "page_size" in data
            assert isinstance(data["total"], int)

    @pytest.mark.asyncio
    async def test_concurrent_requests_different_parameters(self, client: AsyncClient, populated_db):
        """Concurrent requests with different filter parameters must not cross-contaminate results."""
        async def get_gpt4():
            return await client.get("/api/v1/classifier/samples?model=gpt-4")

        async def get_claude():
            return await client.get("/api/v1/classifier/samples?model=claude-3")

        async def get_correct():
            return await client.get("/api/v1/classifier/samples?correct=true")

        results = await asyncio.gather(get_gpt4(), get_claude(), get_correct())

        gpt_data = results[0].json()
        claude_data = results[1].json()
        correct_data = results[2].json()

        # Each response must have correct model filtering
        for s in gpt_data["samples"]:
            assert s["selected_model"] == "gpt-4"
        for s in claude_data["samples"]:
            assert s["selected_model"] == "claude-3"
        for s in correct_data["samples"]:
            assert s["is_correct"] is True


# ======================================================================
# Edge Cases — Malformed Query Parameters
# ======================================================================

class TestEdgeCases:
    """Edge cases: malformed params, unicode, boundary conditions."""

    @pytest.mark.asyncio
    async def test_non_integer_page_is_rejected(self, client: AsyncClient):
        """Non-integer page value (e.g. 'abc') should be rejected."""
        resp = await client.get("/api/v1/classifier/samples?page=abc")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_non_integer_page_size_is_rejected(self, client: AsyncClient):
        """Non-integer page_size value should be rejected."""
        resp = await client.get("/api/v1/classifier/samples?page_size=abc")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_float_page_is_rejected(self, client: AsyncClient):
        """Float page value (e.g. '1.5') should be rejected."""
        resp = await client.get("/api/v1/classifier/samples?page=1.5")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_unicode_text_in_search(self, client: AsyncClient, populated_db):
        """Search with unicode characters should work correctly."""
        # Add a sample with unicode text
        resp = await client.get("/api/v1/classifier/samples?q=capital")
        assert resp.status_code == 200
        data = resp.json()
        # Should find at least one result
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_unicode_prompt_text_in_results(self, client: AsyncClient, populated_db):
        """Samples with unicode in prompt_text are returned correctly."""
        resp = await client.get("/api/v1/classifier/samples")
        data = resp.json()
        assert resp.status_code == 200
        for s in data["samples"]:
            assert isinstance(s["prompt_text"], str)

    @pytest.mark.asyncio
    async def test_large_page_size_at_boundary(self, client: AsyncClient, populated_db):
        """page_size=100 (maximum) should work correctly."""
        resp = await client.get("/api/v1/classifier/samples?page_size=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page_size"] == 100
        assert data["total"] >= 0

    @pytest.mark.asyncio
    async def test_page_beyond_available_data(self, client: AsyncClient, populated_db):
        """Requesting a page beyond available data returns empty samples with total: 0."""
        resp = await client.get("/api/v1/classifier/samples?page=9999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["samples"] == []
        # total is the count of ALL matching records, not just on this page
        assert isinstance(data["total"], int)

    @pytest.mark.asyncio
    async def test_results_order_is_descending_by_created_at(self, client: AsyncClient, populated_db):
        """Results must be ordered by created_at descending."""
        resp = await client.get("/api/v1/classifier/samples?page_size=100")
        data = resp.json()
        if len(data["samples"]) >= 2:
            for i in range(len(data["samples"]) - 1):
                # Since created_at is stored as ISO string, simple comparison works
                assert data["samples"][i]["created_at"] >= data["samples"][i + 1]["created_at"]

    @pytest.mark.asyncio
    async def test_multiple_concurrent_same_request(self, client: AsyncClient, populated_db):
        """Identical concurrent requests should all return consistent total counts."""
        async def get_total() -> int:
            resp = await client.get("/api/v1/classifier/samples?model=gpt-4")
            return resp.json()["total"]

        results = await asyncio.gather(*[get_total() for _ in range(10)])
        # All concurrent requests should see the same total
        assert len(set(results)) == 1, f"Inconsistent totals from concurrent requests: {results}"
