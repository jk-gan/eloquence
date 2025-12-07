<p align="center">
  <img alt="Eloquence" src=".github/media/eloquence.jpg" width="650">
  <h1 align="center">
    Eloquence: Handmade Vector DB from scratch
  </h1>
</p>

<p align="center">A learning project to build a vector database from scratch using <span style="font-weight: bold">Zig</span> — exploring SIMD operations, similarity search algorithms, and database internals.</p>

## Overview

Eloquence is a personal journey into understanding how vector databases work under the hood. Instead of using existing solutions like Pinecone, Milvus, or Weaviate, this project builds core vector DB functionality from first principles.

### Why Zig?

- **SIMD First-Class Support**: Zig's `@Vector` type maps directly to CPU SIMD registers, enabling high-performance vector operations without explicit intrinsics
- **Comptime Powers**: Compile-time computation allows for dimension-specific optimizations without runtime overhead
- **No Hidden Control Flow**: Perfect for understanding exactly what's happening at the hardware level
- **Learning Challenge**: Building something non-trivial in a systems language is the best way to learn it

## Current Features

- ✅ **In-memory vector storage** with dynamic capacity
- ✅ **Multiple distance metrics**:
  - **Cosine Similarity** — Vectors are normalized, then dot product is computed
  - **Dot Product** — Raw dot product without normalization
  - **Euclidean Distance** — L2 distance (negated for max-heap compatibility)
- ✅ **Top-K nearest neighbor search** using a priority queue
- ✅ **Compile-time dimension specification** for zero-cost abstractions

## Quick Start

### Prerequisites

- [Zig](https://ziglang.org/download/) 0.15.2 or later

### Build & Run

```bash
# Build the project
zig build

# Run the demo
zig build run
```

### Example Output

```
Inserting 5,000 random vectors...
Searching top 10 neighbors...
Rank 1 → id: 4000, score: 0.999998
Rank 2 → id: 2847, score: 0.142853
Rank 3 → id: 1923, score: 0.139421
...
```

## Usage

```zig
const std = @import("std");

// Import or define VectorDB...

pub fn main() !void {
    const dim = 128;  // Vector dimension
    const DB = VectorDB(dim, .Cosine);  // Choose your distance metric

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var db = DB.init(allocator);
    defer db.deinit();

    // Add vectors
    const my_vector: @Vector(dim, f32) = // ... your vector data
    try db.add(1, my_vector);

    // Search for similar vectors
    const results = try db.search(query_vector, 10);  // Top 10
    defer allocator.free(results);

    for (results) |result| {
        std.debug.print("ID: {}, Score: {d:.4}\n", .{ result.id, result.score });
    }
}
```

## Roadmap

### Phase 1: Foundation ✅
- [x] Basic vector storage
- [x] Cosine similarity search
- [x] Multiple distance metrics
- [ ] Persistence (save/load to disk)
- [ ] Metadata storage & filtering

### Phase 2: Indexing (The Heart of Vector DBs)
- [ ] **IVF** (Inverted File Index) — Clustering-based ANN
- [ ] **HNSW** (Hierarchical Navigable Small World) — Graph-based ANN
- [ ] **Product Quantization** (PQ) — Vector compression

### Phase 3: Production Features
- [ ] Batch operations
- [ ] Multi-threading
- [ ] Memory-mapped files
- [ ] API layer (HTTP/gRPC)

## How It Works

### Vector Storage

Vectors are stored as Zig's native `@Vector(dim, f32)` type, which maps directly to SIMD registers when possible. This allows operations like dot products to be computed with a single instruction:

```zig
fn dot(comptime dim: usize, a: @Vector(dim, f32), b: @Vector(dim, f32)) f32 {
    return @reduce(.Add, a * b);  // SIMD multiply + horizontal add
}
```

### Cosine Similarity Optimization

For cosine similarity, vectors are normalized at insert time. This transforms the expensive per-query operation:

```
cosine(a, b) = dot(a, b) / (||a|| × ||b||)
```

Into a simple dot product (since normalized vectors have unit length):

```
cosine(a, b) = dot(a, b)  ← Much faster!
```

### Top-K Search

A priority dequeue (min-heap) is used to maintain the K highest-scoring results efficiently, avoiding the need to sort all vectors.

## Learning Resources

Resources that helped build this project:

- [What is a Vector Database?](https://www.pinecone.io/learn/vector-database/)
- [HNSW Algorithm Explained](https://www.pinecone.io/learn/hnsw/)
- [Zig Language Reference](https://ziglang.org/documentation/master/)
- [Zig SIMD and Vectors](https://ziglang.org/documentation/master/#Vectors)

## License

This is a learning project. Feel free to use it as a reference for your own exploration!

---

*"The best way to learn how something works is to build it yourself."*
