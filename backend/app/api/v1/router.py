from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.core.database import get_db
from app.core.models import (
    ClassifierStatus,
    LiveMetric,
    ModelRegistryCreate,
    ModelRegistryEntry,
    ModelRegistryUpdate,
    QueueStatus,
)
from app.models.db import ClassifierSample, Model, RequestLog
from app.services.redis_queue import redis_queue
from app.services.router_service import extract_features, get_model_by_id, get_models

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


@router.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/api/v1/models")
async def list_models(db=Depends(get_db)):
    models = await get_models(db)
    return {"models": [m.to_dict() for m in models], "total": len(models)}


@router.post("/api/v1/models", status_code=201)
async def create_model(entry: ModelRegistryCreate, db=Depends(get_db)):
    existing = await get_model_by_id(db, entry.id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Model '{entry.id}' already exists")

    from app.core.security import encrypt_api_key
    api_key_encrypted = encrypt_api_key(entry.api_key) if entry.api_key else None

    model = Model(
        id=entry.id,
        display_name=entry.display_name,
        provider=entry.provider.value,
        base_url=entry.base_url,
        api_key_encrypted=api_key_encrypted,
        capabilities=[c.value for c in entry.capabilities],
        tags=entry.tags,
        cost_per_1k_input=entry.cost_per_1k_input,
        cost_per_1k_output=entry.cost_per_1k_output,
        max_tokens=entry.max_tokens,
        context_window=entry.context_window,
        rpm_limit=entry.rpm_limit,
        tpm_limit=entry.tpm_limit,
        is_active=entry.is_active,
        priority=entry.priority,
    )
    db.add(model)
    await db.commit()
    await db.refresh(model)
    logger.info("Created model: %s", model.id)
    return model.to_dict()


@router.put("/api/v1/models/{model_id}")
async def update_model(model_id: str, update: ModelRegistryUpdate, db=Depends(get_db)):
    model = await get_model_by_id(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    update_data = update.model_dump(exclude_unset=True)
    if "api_key" in update_data and update_data["api_key"] is not None:
        from app.core.security import encrypt_api_key
        model.api_key_encrypted = encrypt_api_key(update_data.pop("api_key"))
    if "provider" in update_data and update_data["provider"] is not None:
        update_data["provider"] = update_data["provider"].value
    if "capabilities" in update_data and update_data["capabilities"] is not None:
        update_data["capabilities"] = [c.value for c in update_data["capabilities"]]

    for field, value in update_data.items():
        if hasattr(model, field):
            setattr(model, field, value)

    await db.commit()
    await db.refresh(model)
    logger.info("Updated model: %s", model_id)
    return model.to_dict()


@router.delete("/api/v1/models/{model_id}")
async def delete_model(model_id: str, db=Depends(get_db)):
    model = await get_model_by_id(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    await db.delete(model)
    await db.commit()
    logger.info("Deleted model: %s", model_id)
    return {"deleted": model_id}


@router.get("/api/v1/models/{model_id}")
async def get_model(model_id: str, db=Depends(get_db)):
    model = await get_model_by_id(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return model.to_dict()


@router.get("/api/v1/logs")
async def get_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    model_id: str | None = None,
    is_error: bool | None = None,
    db=Depends(get_db),
):
    query = select(RequestLog).order_by(RequestLog.created_at.desc())
    if model_id:
        query = query.where(RequestLog.model_id == model_id)
    if is_error is not None:
        query = query.where(RequestLog.is_error == is_error)
    total_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    logs = list(result.scalars().all())
    return {"logs": [l.to_dict() for l in logs], "total": total, "skip": skip, "limit": limit}


@router.get("/api/v1/metrics/summary")
async def metrics_summary(
    period_minutes: int = Query(60, ge=5, le=1440),
    db=Depends(get_db),
):
    since = datetime.utcnow() - timedelta(minutes=period_minutes)
    result = await db.execute(
        select(
            RequestLog.model_id,
            func.count(RequestLog.id).label("total_requests"),
            func.coalesce(func.sum(RequestLog.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(RequestLog.cost), 0).label("total_cost"),
            func.coalesce(func.avg(RequestLog.latency_ms), 0).label("avg_latency_ms"),
            func.sum(RequestLog.is_error.cast(type(1))).label("error_count"),
        ).where(RequestLog.created_at >= since).group_by(RequestLog.model_id)
    )
    rows = result.all()
    summary = []
    for row in rows:
        summary.append({
            "model_id": row.model_id,
            "total_requests": int(row.total_requests),
            "total_tokens": int(row.total_tokens),
            "total_cost": float(row.total_cost),
            "avg_latency_ms": float(row.avg_latency_ms),
            "error_count": int(row.error_count or 0),
            "period_seconds": period_minutes * 60,
        })
    return {"metrics": summary, "period_minutes": period_minutes}


@router.get("/api/v1/metrics/time-series")
async def metrics_time_series(
    period_minutes: int = Query(60, ge=5, le=1440),
    granularity_minutes: int = Query(5, ge=1, le=60),
    db=Depends(get_db),
):
    since = datetime.utcnow() - timedelta(minutes=period_minutes)
    result = await db.execute(
        select(
            func.date_trunc("hour", RequestLog.created_at).label("bucket"),
            RequestLog.model_id,
            func.count(RequestLog.id).label("requests"),
            func.coalesce(func.avg(RequestLog.latency_ms), 0).label("avg_latency"),
            func.coalesce(func.sum(RequestLog.cost), 0).label("cost"),
            func.sum(RequestLog.is_error.cast(type(1))).label("errors"),
        ).where(RequestLog.created_at >= since)
        .group_by("bucket", RequestLog.model_id)
        .order_by("bucket")
    )
    rows = result.all()
    series = []
    for row in rows:
        series.append({
            "timestamp": row.bucket.isoformat() if row.bucket else "",
            "model_id": row.model_id,
            "requests": int(row.requests),
            "avg_latency_ms": float(row.avg_latency),
            "cost": float(row.cost),
            "errors": int(row.errors or 0),
        })
    return {"time_series": series, "period_minutes": period_minutes, "granularity_minutes": granularity_minutes}


@router.get("/api/v1/metrics/live")
async def live_metrics(db=Depends(get_db)):
    async def event_generator():
        while True:
            try:
                since_5min = datetime.utcnow() - timedelta(minutes=5)
                result = await db.execute(
                    select(
                        func.count(RequestLog.id).label("total"),
                        func.coalesce(func.sum(RequestLog.cost), 0).label("total_cost"),
                        func.coalesce(func.avg(RequestLog.latency_ms), 0).label("avg_latency"),
                        func.sum(RequestLog.is_error.cast(type(1))).label("errors"),
                    ).where(RequestLog.created_at >= since_5min)
                )
                row = result.one()
                total = int(row.total)
                error_count = int(row.errors or 0)
                avg_lat = round(float(row.avg_latency), 2)

                queue_depth = 0
                try:
                    queue_depth = await redis_queue.queue_depth()
                except Exception:
                    pass

                top_result = await db.execute(
                    select(RequestLog.model_id, func.count(RequestLog.id).label("cnt"))
                    .where(RequestLog.created_at >= since_5min)
                    .group_by(RequestLog.model_id)
                    .order_by(func.count(RequestLog.id).desc())
                    .limit(1)
                )
                top_row = top_result.first()
                top_model = top_row.model_id if top_row else ""

                metric = LiveMetric(
                    request_rate=round(total / 300, 2) if total > 0 else 0.0,
                    active_requests=total % 100,
                    queue_depth=queue_depth,
                    avg_latency_ms=avg_lat,
                    error_rate=round(error_count / max(total, 1), 4),
                    total_requests=total,
                    total_cost=float(row.total_cost),
                    top_model=top_model,
                    timestamp=datetime.utcnow().isoformat(),
                )
                yield {"event": "metric", "data": metric.model_dump_json()}
            except Exception:
                logger.exception("Error streaming live metrics")
                yield {"event": "error", "data": json.dumps({"error": "metrics error"})}
            await asyncio.sleep(5)

    return EventSourceResponse(event_generator())


@router.get("/api/v1/classifier")
async def classifier_status(db=Depends(get_db)):
    result = await db.execute(
        select(
            func.count(ClassifierSample.id).label("total"),
        )
    )
    total = result.scalar() or 0

    correct_result = await db.execute(
        select(func.count(ClassifierSample.id))
        .where(ClassifierSample.is_correct == True)
    )
    correct = correct_result.scalar() or 0

    latest_result = await db.execute(
        select(ClassifierSample.created_at)
        .order_by(ClassifierSample.created_at.desc())
        .limit(1)
    )
    latest = latest_result.scalar()

    accuracy = round(correct / max(total, 1), 4) if total > 0 else None

    return ClassifierStatus(
        model_version="v1.0",
        accuracy=accuracy,
        training_data_count=total,
        last_trained_at=latest,
        is_training=False,
    ).model_dump()


@router.get("/api/v1/queue")
async def queue_status():
    depth = 0
    in_flight = 0
    try:
        depth = await redis_queue.queue_depth()
        in_flight = await redis_queue.in_flight_count()
    except Exception:
        logger.exception("Failed to get queue depth")
    return QueueStatus(
        depth=depth,
        workers_active=in_flight,
        avg_processing_time_ms=0.0,
        consumed_total=0,
        failed_total=0,
    ).model_dump()


@router.get("/api/v1/metrics/dashboard")
async def dashboard_metrics(db=Depends(get_db)):
    since_24h = datetime.utcnow() - timedelta(hours=24)

    total_result = await db.execute(
        select(func.count(RequestLog.id)).where(RequestLog.created_at >= since_24h)
    )
    total_requests = total_result.scalar() or 0

    cost_result = await db.execute(
        select(func.coalesce(func.sum(RequestLog.cost), 0))
        .where(RequestLog.created_at >= since_24h)
    )
    total_cost = float(cost_result.scalar() or 0)

    tokens_result = await db.execute(
        select(func.coalesce(func.sum(RequestLog.total_tokens), 0))
        .where(RequestLog.created_at >= since_24h)
    )
    total_tokens = int(tokens_result.scalar() or 0)

    errors_result = await db.execute(
        select(func.count(RequestLog.id))
        .where(RequestLog.created_at >= since_24h, RequestLog.is_error == True)
    )
    error_count = errors_result.scalar() or 0

    top_models_result = await db.execute(
        select(RequestLog.model_id, func.count(RequestLog.id).label("cnt"))
        .where(RequestLog.created_at >= since_24h)
        .group_by(RequestLog.model_id)
        .order_by(func.count(RequestLog.id).desc())
        .limit(10)
    )
    top_models = [
        {"model_id": row.model_id, "count": int(row.cnt)}
        for row in top_models_result.all()
    ]

    hourly_result = await db.execute(
        select(
            func.date_trunc("hour", RequestLog.created_at).label("bucket"),
            func.count(RequestLog.id).label("cnt"),
            func.coalesce(func.avg(RequestLog.latency_ms), 0).label("avg_lat"),
            func.coalesce(func.sum(RequestLog.cost), 0).label("cost_sum"),
        )
        .where(RequestLog.created_at >= since_24h)
        .group_by("bucket")
        .order_by("bucket")
    )
    hourly = [
        {
            "timestamp": row.bucket.isoformat() if row.bucket else "",
            "requests": int(row.cnt),
            "avg_latency_ms": float(row.avg_lat),
            "cost": float(row.cost_sum),
        }
        for row in hourly_result.all()
    ]

    return {
        "total_requests": total_requests,
        "total_cost": round(total_cost, 4),
        "total_tokens": total_tokens,
        "error_count": error_count,
        "error_rate": round(error_count / max(total_requests, 1), 4),
        "top_models": top_models,
        "hourly": hourly,
    }
