# Mermaid Diagrams

Reference diagrams for `architecture.md` and `systemDesign.md`. Paste any block into
a Mermaid renderer (or GitHub, which renders these natively in `.md` files).

## 1. Module / Crate Architecture

```mermaid
flowchart LR
    subgraph storage["storage/"]
        Column
        Table
        CSVLoader["CSV loader"]
    end

    subgraph parser["parser/"]
        SQLParser["sqlparser-rs AST"]
    end

    subgraph plan["plan/"]
        LogicalPlan
        OperatorTreeBuilder["AST -> Operator tree"]
    end

    subgraph exec["exec/"]
        Scan
        Filter
        Project
        Aggregate
        RecordBatch
    end

    subgraph simdmod["simd/"]
        SIMDFilter["SIMD filter compare"]
        SIMDSum["SIMD sum reduction"]
    end

    subgraph bench["bench/"]
        NaiveBaseline["naive row-scan"]
        Criterion["criterion benchmarks"]
    end

    CSVLoader --> Table
    Table --> Column
    SQLParser --> LogicalPlan
    LogicalPlan --> OperatorTreeBuilder
    OperatorTreeBuilder --> Scan
    Scan --> Filter --> Project --> Aggregate
    Scan -.uses.-> RecordBatch
    Filter -.uses.-> RecordBatch
    Aggregate -.uses.-> RecordBatch
    Filter -.hot loop.-> SIMDFilter
    Aggregate -.hot loop.-> SIMDSum
    NaiveBaseline -.compared against.-> Aggregate
    Criterion --> NaiveBaseline
    Criterion --> Aggregate
```

## 2. Query Execution Data Flow

```mermaid
flowchart TD
    A[CSV file] --> B["storage::Table (columnar, in memory)"]
    C["SQL string"] --> D["sqlparser AST"]
    D --> E["LogicalPlan"]
    E --> F["Operator tree: Scan -> Filter -> Project -> Aggregate"]
    B --> F
    F --> G["Scan: read Table in ~1024-row batches"]
    G --> H["Filter: compute bitset, compact batch"]
    H --> I["Project: select columns"]
    I --> J["Aggregate: build group index, then scatter-accumulate"]
    J --> K["Result Table: one row per group"]
```

## 3. Operator Pull Chain (sequence view)

```mermaid
sequenceDiagram
    participant Agg as Aggregate
    participant Proj as Project
    participant Filt as Filter
    participant Scan as Scan
    participant Tbl as storage::Table

    Agg->>Proj: next_batch()
    Proj->>Filt: next_batch()
    Filt->>Scan: next_batch()
    Scan->>Tbl: read next ~1024 rows
    Tbl-->>Scan: RecordBatch (raw)
    Scan-->>Filt: RecordBatch (raw)
    Filt->>Filt: compute bitset over batch
    Filt->>Filt: compact -> new RecordBatch (filtered rows only)
    Filt-->>Proj: RecordBatch (filtered)
    Proj->>Proj: select columns
    Proj-->>Agg: RecordBatch (projected)
    Agg->>Agg: phase 1: build group index
    Agg->>Agg: phase 2: scatter-accumulate into Accumulator per slot
    Agg-->>Agg: repeat until Scan exhausted
    Agg-->>Agg: finalize() all accumulators -> result Table
```

## 4. Hash Group-By — Two-Phase Detail

```mermaid
flowchart TD
    A["RecordBatch (col1: group col, col2: sum col)"] --> B["Phase 1: Build group index"]
    B --> B1["for each row: hash GroupKey(code) -> lookup/insert slot in HashMap"]
    B1 --> C["group_key -> accumulator slot mapping"]
    A --> D["Phase 2: Scatter-accumulate"]
    C --> D
    D --> D1["for each row: accumulators[slot].update(col2[row])"]
    D1 --> E["Per-group SumAccumulator totals"]

    B1 -.SIMD note.-> N1["Sequential read of codes\npartially vectorizes on read side"]
    D1 -.SIMD note.-> N2["Data-dependent slot index\ndefeats vectorization entirely"]
```

## 5. Filter: Compaction vs. Selection-Vector Passthrough (tradeoff)

```mermaid
flowchart LR
    subgraph chosen["Chosen: Compaction"]
        A1["RecordBatch in"] --> A2["compute bitset"]
        A2 --> A3["materialize new, smaller RecordBatch\n(selected rows only)"]
        A3 --> A4["Downstream operators see plain batches,\nnever know filtering happened"]
    end

    subgraph alt["Not chosen: Selection-vector passthrough"]
        B1["RecordBatch in"] --> B2["compute selection vector"]
        B2 --> B3["pass (batch, selection vector) downstream"]
        B3 --> B4["every downstream operator must\napply the selection vector itself"]
    end

    A4 -.tradeoff.-> T1["Cost: one copy per filtered batch"]
    B4 -.tradeoff.-> T2["Cost: couples every operator's\nsignature to filtered execution"]
```

## 6. SIMD Applicability Map

```mermaid
flowchart TD
    Start["Hot loop candidate"] --> Q1{"Dense numeric op?\n(Int64/Float64)"}
    Q1 -->|Yes| Q2{"Output index\ndata-independent?"}
    Q1 -->|No, it's Utf8| NoSIMD1["No SIMD\n(scalar dict comparison)"]
    Q2 -->|Yes: filter compare, sum reduction| YesSIMD["SIMD applied"]
    Q2 -->|No: group-by scatter-accumulate| NoSIMD2["No SIMD\n(data-dependent scatter write)"]
```

## 7. Core Type Relationships

```mermaid
classDiagram
    class Column {
        <<enum>>
        Int64(Vec~i64~)
        Float64(Vec~f64~)
        Utf8Dict
    }
    class RecordBatch {
        HashMap~String, Column~ columns
        usize len
    }
    class GroupKey {
        u64 code
    }
    class Accumulator~T~ {
        <<trait>>
        %% confirmed 2026-08-18: generic over T — see systemDesign.md "Accumulator"
        +update(val: T)
        +finalize() T
    }
    class SumAccumulator~T~ {
        T total
        +update(val: T)
        +finalize() T
    }
    class Operator {
        <<trait>>
        +next_batch() Option~RecordBatch~
    }
    class Scan
    class Filter
    class Project
    class Aggregate

    Operator <|.. Scan
    Operator <|.. Filter
    Operator <|.. Project
    Operator <|.. Aggregate
    Accumulator <|.. SumAccumulator
    Aggregate --> GroupKey : keys on
    Aggregate --> Accumulator : one per group slot
    RecordBatch --> Column : contains
    Scan --> RecordBatch : emits
    Filter --> RecordBatch : emits (compacted)
```

## 8. Benchmark Comparison Structure

```mermaid
flowchart TD
    Data["Synthetic CSV, few million rows"] --> Naive["Naive row-scan\n(no operators, no batching)"]
    Data --> Batched["Batched / vectorized\n(RecordBatch pipeline, hash group-by)"]
    Data --> SIMDv["Batched + SIMD\n(filter compare, sum reduction)"]
    Data --> SortG["Sort-based grouping\n(secondary comparison)"]

    Naive --> Report["Timing table + chart"]
    Batched --> Report
    SIMDv --> Report
    SortG --> Report

    Batched -.profiled separately.-> P1["Phase 1: build group index"]
    Batched -.profiled separately.-> P2["Phase 2: scatter-accumulate"]
    P1 --> Report
    P2 --> Report
```

## 9. Build Phases (flow)

```mermaid
flowchart LR
    P0["Phase 0\nSetup"] --> P1["Phase 1\nStorage layer"]
    P1 --> P2["Phase 2\nSQL parsing"]
    P2 --> P3["Phase 3\nNaive baseline"]
    P3 --> P4["Phase 4\nBatch execution pipeline"]
    P4 --> P5["Phase 5\nSIMD"]
    P5 --> P6["Phase 6\nSort-based grouping + benchmarks"]
    P6 --> P7["Phase 7\nWrite-up"]

    P3 -.correctness reference for.-> P4
    P4 -.scalar version required before.-> P5
```
