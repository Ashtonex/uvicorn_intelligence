from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any
import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

@dataclass(frozen=True)
class NirvanaFrames:
    sales: pd.DataFrame
    ledger: pd.DataFrame
    inventory: pd.DataFrame
    allocations: pd.DataFrame
    shops: pd.DataFrame
    employees: pd.DataFrame
    operations: pd.DataFrame

def get_client() -> Client:
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
    if not url or not key:
        raise RuntimeError("Missing Supabase env: NEXT_PUBLIC_SUPABASE_URL and a Supabase key are required.")
    return create_client(url, key)

def _coerce_dates(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
    return df

def _coerce_numbers(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df

def load_table(table: str, columns: str = "*", limit: int = 50000, order_by: str | None = None) -> pd.DataFrame:
    client = get_client()
    query = client.table(table).select(columns)
    if order_by:
        query = query.order(order_by, desc=True)
    response = query.limit(limit).execute()
    return pd.DataFrame(response.data or [])

def load_sales(limit: int = 50000) -> pd.DataFrame:
    df = load_table("sales", "*", limit=limit, order_by="date")
    return _coerce_numbers(_coerce_dates(df, ["date", "created_at"]), ["quantity", "unit_price", "total_with_tax", "total_before_tax", "tax"])

def load_ledger(limit: int = 50000) -> pd.DataFrame:
    df = load_table("ledger_entries", "*", limit=limit, order_by="date")
    return _coerce_numbers(_coerce_dates(df, ["date", "created_at"]), ["amount"])

def load_inventory(limit: int = 50000) -> pd.DataFrame:
    df = load_table("inventory_items", "*", limit=limit)
    return _coerce_numbers(_coerce_dates(df, ["date_added", "created_at", "updated_at"]), ["quantity", "landed_cost", "cost", "price", "selling_price"])

def load_allocations(limit: int = 50000) -> pd.DataFrame:
    df = load_table("inventory_allocations", "*", limit=limit)
    return _coerce_numbers(df, ["quantity"])

def load_operations(limit: int = 50000) -> pd.DataFrame:
    df = load_table("operations_ledger", "*", limit=limit, order_by="created_at")
    return _coerce_numbers(_coerce_dates(df, ["created_at", "effective_date"]), ["amount"])

def load_core_frames(limit: int = 50000) -> NirvanaFrames:
    return NirvanaFrames(
        sales=load_sales(limit),
        ledger=load_ledger(limit),
        inventory=load_inventory(limit),
        allocations=load_allocations(limit),
        shops=load_table("shops", "*", limit=limit),
        employees=load_table("employees", "*", limit=limit),
        operations=load_operations(limit),
    )

def save_analytics_result(kind: str, payload: dict, summary: str | None = None, status: str = "success") -> dict:
    client = get_client()
    row = {
        "kind": kind,
        "status": status,
        "generated_at": payload.get("generated_at"),
        "summary": summary,
        "payload": payload,
    }
    response = client.table("analytics_results").insert(row).execute()
    return (response.data or [{}])[0]
