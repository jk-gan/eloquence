const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

pub fn KMeans(comptime dim: usize) type {
    return struct {
        const Vec = @Vector(dim, f32);
        const Self = @This();

        k: usize,
        max_iter: usize,
        tolerance: f32,

        centroids: ?[]Vec,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, k: usize, max_iter: usize, tolerance: f32) Self {
            return .{
                .k = k,
                .max_iter = max_iter,
                .tolerance = tolerance,
                .centroids = null,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.centroids) |centroids| {
                self.allocator.free(centroids);
                self.centroids = null;
            }
        }

        pub fn train(self: *Self, vectors: []const Vec) !void {
            if (vectors.len < self.k) return error.NotEnoughVectors;

            if (self.centroids) |old| self.allocator.free(old);
            const centroids = try self.allocator.alloc(Vec, self.k);
            self.centroids = centroids;

            var prng = std.Random.DefaultPrng.init(112233);
            const random = prng.random();

            // vectors = [v₀, v₁, v₂, v₃, v₄]   k = 3
            //
            // Step 1: Pick v₂ randomly
            //         centroids = [v₂]
            //
            // Step 2: Compute D² (distance to v₂):
            //         D² = [4.0, 1.0, 0.0, 9.0, 16.0]
            //               │     │    │     │     └── v₄ is far (16)
            //               │     │    │     └── v₃ is far (9)
            //               │     │    └── v₂ is the centroid (0)
            //               │     └── v₁ is close (1)
            //               └── v₀ is medium (4)
            //
            //         total = 4 + 1 + 0 + 9 + 16 = 30
            //         threshold = random(0, 30) → say 22
            //
            //         cumulative: 4 → 5 → 5 → 14 → 30
            //                                    ↑
            //                     14 < 22, but 30 ≥ 22 → pick v₄!
            //
            //         centroids = [v₂, v₄]
            //
            // Step 3: Compute D² (distance to NEAREST of v₂ or v₄):
            //         D²[i] = min(dist(vᵢ, v₂), dist(vᵢ, v₄))
            //
            //         D² = [3.0, 1.0, 0.0, 2.0, 0.0]
            //         total = 6
            //         threshold = random(0, 6) → say 4.5
            //
            //         cumulative: 3 → 4 → 4 → 6
            //                               ↑
            //                     4 < 4.5, but 6 ≥ 4.5 → pick v₃!
            //
            //         centroids = [v₂, v₄, v₃]
            //
            // Done! 3 well-spread centroids.
            const min_distances_squared = try self.allocator.alloc(f32, vectors.len);
            defer self.allocator.free(min_distances_squared);

            const first_idx = random.intRangeLessThan(usize, 0, vectors.len);
            centroids[0] = vectors[first_idx];

            for (1..self.k) |i| {
                var total: f32 = 0.0;

                for (vectors, 0..) |vec, j| {
                    var min_distance_squared: f32 = std.math.floatMax(f32);
                    for (centroids[0..i]) |centroid| {
                        const d = distance_squared(vec, centroid);
                        min_distance_squared = @min(min_distance_squared, d);
                    }
                    min_distances_squared[j] = min_distance_squared;
                    total += min_distance_squared;
                }

                const threshold = random.float(f32) * total;
                var cumulative: f32 = 0.0;
                var selected: usize = vectors.len - 1; // fallback to the last vector
                for (min_distances_squared, 0..) |d_sq, k| {
                    cumulative += d_sq;
                    if (cumulative >= threshold) {
                        selected = k;
                        break;
                    }
                }

                centroids[i] = vectors[selected];
            }

            var assignments = try self.allocator.alloc(usize, vectors.len);
            defer self.allocator.free(assignments);

            var sums = try self.allocator.alloc(Vec, self.k);
            defer self.allocator.free(sums);

            var counts = try self.allocator.alloc(usize, self.k);
            defer self.allocator.free(counts);

            const old_centroids = try self.allocator.alloc(Vec, self.k);
            defer self.allocator.free(old_centroids);

            for (0..self.max_iter) |_| {
                for (vectors, 0..) |vec, i| {
                    var best_cluster: usize = 0;
                    var best_distance: f32 = distance_squared(vec, centroids[0]);

                    for (centroids[1..], 1..) |centroid, j| {
                        const distance = distance_squared(vec, centroid);
                        if (distance < best_distance) {
                            best_distance = distance;
                            best_cluster = j;
                        }
                    }
                    assignments[i] = best_cluster;
                }

                @memcpy(old_centroids, centroids);

                for (sums) |*s| s.* = zero_vec();
                for (counts) |*c| c.* = 0;

                for (vectors, assignments) |vec, cluster| {
                    sums[cluster] += vec;
                    counts[cluster] += 1;
                }

                for (centroids, sums, counts) |*c, sum, count| {
                    if (count > 0) {
                        c.* = sum / @as(Vec, @splat(@floatFromInt(count)));
                    } else {
                        // Empty cluster: reinitialize from random vector
                        c.* = vectors[random.intRangeLessThan(usize, 0, vectors.len)];
                    }
                }

                var total_movement: f32 = 0.0;
                for (centroids, old_centroids) |new, old| {
                    total_movement += distance_squared(new, old);
                }

                if (total_movement < self.tolerance) break;
            }
        }

        // ┌─────────────────────────────────────────────────────────────┐
        // │  predict(vector) → cluster index                            │
        // │                                                             │
        // │  Input:  query vector Q                                     │
        // │  Output: index of nearest centroid (0 to k-1)               │
        // │                                                             │
        // │          C₀                                                 │
        // │             \                                               │
        // │              \  d=2.1                                       │
        // │               \                                             │
        // │                Q ←── query                                  │
        // │               /                                             │
        // │              /  d=0.8  ← smallest!                          │
        // │             /                                               │
        // │          C₁                                                 │
        // │                                                             │
        // │  → returns 1 (index of C₁)                                  │
        // └─────────────────────────────────────────────────────────────┘
        pub fn predict(self: *const Self, vector: Vec) usize {
            const centroids = self.centroids.?;

            var best: usize = 0;
            var best_distance = distance_squared(vector, centroids[0]);

            for (centroids[1..], 1..) |centroid, i| {
                const distance = distance_squared(vector, centroid);
                if (distance < best_distance) {
                    best_distance = distance;
                    best = i;
                }
            }

            return best;
        }

        fn distance_squared(a: Vec, b: Vec) f32 {
            const diff = a - b;
            return @reduce(.Add, diff * diff);
        }

        fn zero_vec() Vec {
            return @splat(0.0);
        }
    };
}

// ┌─────────────────────────────────────────────────────────────┐
// │  IVFIndex: Inverted File Index for approximate NN search    │
// │                                                             │
// │  Structure:                                                 │
// │    centroids: [C₀, C₁, ..., C_{nlist-1}]                    │
// │    postings:  [vec_indices...]  ← flat, concatenated        │
// │    offsets:   [0, n₀, n₀+n₁, ...]  ← where each list starts │
// │                                                             │
// │  Search flow:                                               │
// │    1. Find nprobe nearest centroids to query                │
// │    2. Scan only those posting lists                         │
// │    3. Return top-k from scanned candidates                  │
// └─────────────────────────────────────────────────────────────┘
pub fn IVFIndex(comptime dim: usize) type {
    return struct {
        const Self = @This();
        pub const Vec = @Vector(dim, f32);
        const KM = KMeans(dim);

        // Trained centroids (owned)
        centroids: []Vec,

        // Flat posting lists: all cluster members concatenated
        // postings[offsets[i]..offsets[i+1]] = vector indices in cluster i
        postings: []u32,
        offsets: []usize,

        // Reference to original vectors (not owned - VectorDB owns these)
        vectors: []const Vec,

        allocator: std.mem.Allocator,

        // ┌─────────────────────────────────────────────────────────────┐
        // │  build() - Construct IVF index from vectors                 │
        // │                                                             │
        // │  Input:  vectors to index, nlist (number of clusters)       │
        // │  Output: IVFIndex ready for search                          │
        // │                                                             │
        // │  Steps:                                                     │
        // │    1. Train KMeans on vectors → centroids                   │
        // │    2. Assign each vector to nearest centroid                │
        // │    3. Count vectors per cluster                             │
        // │    4. Build offsets array (prefix sum of counts)            │
        // │    5. Fill postings array                                   │
        // └─────────────────────────────────────────────────────────────┘
        pub fn build(
            allocator: std.mem.Allocator,
            vectors: []const Vec,
            nlist: usize,
            kmeans_max_iter: usize,
            kmeans_tolerance: f32,
        ) !Self {
            if (vectors.len < nlist) return error.NotEnoughVectors;

            // Step 1: Train KMeans to find cluster centroids
            var kmeans = KM.init(allocator, nlist, kmeans_max_iter, kmeans_tolerance);
            errdefer kmeans.deinit();
            try kmeans.train(vectors);

            // Take ownership of centroids from KMeans
            const centroids = kmeans.centroids.?;
            kmeans.centroids = null; // prevent double-free

            // Step 2: Assign each vector to its nearest centroid
            // assignments[i] = cluster index for vectors[i]
            const assignments = try allocator.alloc(usize, vectors.len);
            defer allocator.free(assignments);

            for (vectors, 0..) |vec, i| {
                assignments[i] = predictCluster(centroids, vec);
            }

            // Step 3: Count how many vectors in each cluster
            // counts[c] = number of vectors assigned to cluster c
            const counts = try allocator.alloc(usize, nlist);
            defer allocator.free(counts);
            @memset(counts, 0);

            for (assignments) |cluster| {
                counts[cluster] += 1;
            }

            // Step 4: Build offsets array using prefix sum
            // offsets[i] = starting position of cluster i in postings array
            //
            // Example: counts = [2, 3, 3]
            //          offsets = [0, 2, 5, 8]
            //                     ↑  ↑  ↑  ↑
            //                    C₀ C₁ C₂ end
            const offsets = try allocator.alloc(usize, nlist + 1);
            errdefer allocator.free(offsets);

            offsets[0] = 0;
            for (counts, 0..) |count, i| {
                offsets[i + 1] = offsets[i] + count;
            }

            // Step 5: Fill postings array
            // Use a copy of offsets as write cursors (we increment as we write)
            //
            // For each vector i with assignment c:
            //   postings[cursor[c]] = i
            //   cursor[c]++
            const postings = try allocator.alloc(u32, vectors.len);
            errdefer allocator.free(postings);

            const cursors = try allocator.alloc(usize, nlist);
            defer allocator.free(cursors);
            @memcpy(cursors, offsets[0..nlist]);

            for (assignments, 0..) |cluster, vec_idx| {
                postings[cursors[cluster]] = @intCast(vec_idx);
                cursors[cluster] += 1;
            }

            return .{
                .centroids = centroids,
                .postings = postings,
                .offsets = offsets,
                .vectors = vectors,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.centroids);
            self.allocator.free(self.postings);
            self.allocator.free(self.offsets);
        }

        // Helper: find nearest centroid for a vector
        fn predictCluster(centroids: []const Vec, vector: Vec) usize {
            var best: usize = 0;
            var best_dist = distanceSquared(vector, centroids[0]);

            for (centroids[1..], 1..) |centroid, i| {
                const dist = distanceSquared(vector, centroid);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = i;
                }
            }
            return best;
        }

        fn distanceSquared(a: Vec, b: Vec) f32 {
            const diff = a - b;
            return @reduce(.Add, diff * diff);
        }

        // ┌─────────────────────────────────────────────────────────────┐
        // │  search() - Find k nearest neighbors using IVF              │
        // │                                                             │
        // │  Query: Q                                                   │
        // │                                                             │
        // │  Step 1: Find nprobe nearest centroids                      │
        // │          ┌───┐                                              │
        // │          │C₀ │ ← dist=5.2                                   │
        // │          └───┘                                              │
        // │              Q ← query                                      │
        // │          ┌───┐                                              │
        // │          │C₁ │ ← dist=1.3 ✓ (closest)                       │
        // │          └───┘                                              │
        // │          ┌───┐                                              │
        // │          │C₂ │ ← dist=2.1 ✓ (2nd closest)                   │
        // │          └───┘                                              │
        // │                                                             │
        // │  Step 2: Scan posting lists of C₁ and C₂                    │
        // │          C₁ → [v₃, v₇, v₁₂]  ← compute dist(Q, each)        │
        // │          C₂ → [v₁, v₅]       ← compute dist(Q, each)        │
        // │                                                             │
        // │  Step 3: Keep top-k closest from all candidates             │
        // └─────────────────────────────────────────────────────────────┘
        pub const SearchResult = struct {
            index: u32, // index into original vectors array
            distance: f32,
        };

        pub fn search(self: *const Self, query: Vec, k: usize, nprobe: usize) ![]SearchResult {
            if (k == 0) return &[_]SearchResult{};

            const actual_nprobe = @min(nprobe, self.centroids.len);

            // Step 1: Find nprobe nearest centroids
            // We'll use a simple array and partial sort for small nprobe
            const centroid_dists = try self.allocator.alloc(CentroidDist, self.centroids.len);
            defer self.allocator.free(centroid_dists);

            for (self.centroids, 0..) |centroid, i| {
                centroid_dists[i] = .{
                    .index = i,
                    .distance = distanceSquared(query, centroid),
                };
            }

            // Partial sort: get nprobe smallest distances
            // For small nprobe, selection is fine; could use heap for large nprobe
            std.mem.sortUnstable(CentroidDist, centroid_dists, {}, compareCentroidDist);

            // Step 2 & 3: Scan posting lists and keep top-k candidates
            // Use a max-heap: if new candidate is better than worst in heap, swap
            const PQ = std.PriorityDequeue(SearchResult, void, compareSearchResult);
            var heap = PQ.init(self.allocator, {});
            defer heap.deinit();

            // Scan the nprobe nearest clusters
            for (centroid_dists[0..actual_nprobe]) |cd| {
                const cluster = cd.index;
                const start = self.offsets[cluster];
                const end = self.offsets[cluster + 1];

                // For each vector in this cluster's posting list
                for (self.postings[start..end]) |vec_idx| {
                    const vec = self.vectors[vec_idx];
                    const dist = distanceSquared(query, vec);

                    const result = SearchResult{ .index = vec_idx, .distance = dist };

                    if (heap.len < k) {
                        try heap.add(result);
                    } else if (dist < heap.peekMax().?.distance) {
                        // New candidate is better than worst in heap
                        _ = heap.removeMax();
                        try heap.add(result);
                    }
                }
            }

            // Extract results in ascending distance order
            var results = try self.allocator.alloc(SearchResult, heap.len);
            var i = heap.len;
            while (heap.removeMinOrNull()) |res| {
                i -= 1;
                results[i] = res;
            }

            return results;
        }

        const CentroidDist = struct {
            index: usize,
            distance: f32,
        };

        fn compareCentroidDist(_: void, a: CentroidDist, b: CentroidDist) bool {
            return a.distance < b.distance;
        }

        fn compareSearchResult(_: void, a: SearchResult, b: SearchResult) std.math.Order {
            return std.math.order(a.distance, b.distance);
        }
    };
}

test "IVFIndex search finds nearest neighbors" {
    const allocator = std.testing.allocator;
    const IVF = IVFIndex(2);

    // Create vectors in two clusters
    const vectors: []const IVF.Vec = &.{
        .{ 0.0, 0.0 }, // 0: cluster A
        .{ 0.1, 0.0 }, // 1: cluster A
        .{ 0.0, 0.1 }, // 2: cluster A
        .{ 10.0, 10.0 }, // 3: cluster B
        .{ 10.1, 10.0 }, // 4: cluster B
        .{ 10.0, 10.1 }, // 5: cluster B
    };

    var index = try IVF.build(allocator, vectors, 2, 10, 1e-4);
    defer index.deinit();

    // Search near cluster A - should find vectors 0, 1, 2
    {
        const results = try index.search(.{ 0.05, 0.05 }, 3, 1);
        defer allocator.free(results);

        try std.testing.expectEqual(3, results.len);

        // All results should be from cluster A (indices 0, 1, 2)
        for (results) |r| {
            try std.testing.expect(r.index <= 2);
        }
    }

    // Search near cluster B with nprobe=2 (search both clusters)
    {
        const results = try index.search(.{ 10.0, 10.0 }, 2, 2);
        defer allocator.free(results);

        try std.testing.expectEqual(2, results.len);
        // Results should be from cluster B (indices 3, 4, 5)
        for (results) |r| {
            try std.testing.expect(r.index >= 3 and r.index <= 5);
        }
    }
}

// ┌─────────────────────────────────────────────────────────────┐
// │  Recall Test: Compare IVF to brute-force ground truth       │
// │                                                             │
// │  Recall = |IVF results ∩ Ground truth| / |Ground truth|    │
// │                                                             │
// │  Example:                                                   │
// │    Ground truth (brute-force top-5): [1, 2, 3, 4, 5]       │
// │    IVF results:                      [1, 2, 3, 7, 8]       │
// │    Intersection: [1, 2, 3] → 3 matches                      │
// │    Recall = 3/5 = 60%                                       │
// └─────────────────────────────────────────────────────────────┘
test "IVFIndex recall vs brute-force" {
    const allocator = std.testing.allocator;
    const IVF = IVFIndex(8);

    // Generate synthetic dataset with clustered structure
    // 100 vectors in 8 dimensions, forming ~4 natural clusters
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const n_vectors: usize = 100;
    const vectors = try allocator.alloc(IVF.Vec, n_vectors);
    defer allocator.free(vectors);

    // Create 4 cluster centers and add noise around them
    const centers: [4]IVF.Vec = .{
        .{ 0, 0, 0, 0, 0, 0, 0, 0 },
        .{ 10, 10, 10, 10, 0, 0, 0, 0 },
        .{ 0, 0, 0, 0, 10, 10, 10, 10 },
        .{ 10, 10, 10, 10, 10, 10, 10, 10 },
    };

    for (0..n_vectors) |i| {
        const center = centers[i % 4];
        var vec: IVF.Vec = undefined;
        for (0..8) |d| {
            vec[d] = center[d] + (random.float(f32) - 0.5) * 2.0; // ±1.0 noise
        }
        vectors[i] = vec;
    }

    // Build IVF index with 4 clusters
    var index = try IVF.build(allocator, vectors, 4, 50, 1e-4);
    defer index.deinit();

    // Test query (near cluster 0)
    const query: IVF.Vec = .{ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    const k: usize = 10;

    // Brute-force ground truth
    const BruteForceResult = struct {
        index: usize,
        distance: f32,
    };

    var ground_truth = try allocator.alloc(BruteForceResult, n_vectors);
    defer allocator.free(ground_truth);

    for (vectors, 0..) |vec, i| {
        const diff = query - vec;
        ground_truth[i] = .{
            .index = i,
            .distance = @reduce(.Add, diff * diff),
        };
    }

    // Sort by distance
    std.mem.sortUnstable(BruteForceResult, ground_truth, {}, struct {
        fn lessThan(_: void, a: BruteForceResult, b: BruteForceResult) bool {
            return a.distance < b.distance;
        }
    }.lessThan);

    // Get ground truth top-k indices
    var truth_set = std.AutoHashMap(u32, void).init(allocator);
    defer truth_set.deinit();
    for (ground_truth[0..k]) |r| {
        try truth_set.put(@intCast(r.index), {});
    }

    // IVF search with different nprobe values
    const nprobe_values = [_]usize{ 1, 2, 4 };
    var recalls: [3]f32 = undefined;

    for (nprobe_values, 0..) |nprobe, np_idx| {
        const ivf_results = try index.search(query, k, nprobe);
        defer allocator.free(ivf_results);

        // Count matches
        var matches: usize = 0;
        for (ivf_results) |r| {
            if (truth_set.contains(r.index)) matches += 1;
        }

        recalls[np_idx] = @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(k));
    }

    // With nprobe=4 (all clusters), should have perfect recall
    try std.testing.expectEqual(@as(f32, 1.0), recalls[2]);

    // Higher nprobe should give equal or better recall
    try std.testing.expect(recalls[1] >= recalls[0]);
    try std.testing.expect(recalls[2] >= recalls[1]);
}

test "IVFIndex build creates correct posting lists" {
    const allocator = std.testing.allocator;
    const IVF = IVFIndex(2);

    // Two clear clusters: around (0,0) and (10,10)
    const vectors: []const IVF.Vec = &.{
        .{ 0.0, 0.0 },
        .{ 0.1, 0.1 },
        .{ 10.0, 10.0 },
        .{ 10.1, 10.1 },
    };

    var index = try IVF.build(allocator, vectors, 2, 10, 1e-4);
    defer index.deinit();

    // Should have 2 centroids
    try std.testing.expectEqual(2, index.centroids.len);

    // Should have 3 offsets: [start_c0, start_c1, end]
    try std.testing.expectEqual(3, index.offsets.len);

    // All 4 vectors should be in postings
    try std.testing.expectEqual(4, index.postings.len);

    // Each cluster should have 2 vectors
    const cluster0_size = index.offsets[1] - index.offsets[0];
    const cluster1_size = index.offsets[2] - index.offsets[1];
    try std.testing.expectEqual(2, cluster0_size);
    try std.testing.expectEqual(2, cluster1_size);
}

test "KMeans basic clustering" {
    const allocator = std.testing.allocator;
    const KM = KMeans(2);

    var vectors = try allocator.alloc(KM.Vec, 10);
    defer allocator.free(vectors);

    // Cluster 1: around (0, 0)
    vectors[0] = .{ 0.0, 0.0 };
    vectors[1] = .{ 0.1, 0.0 };
    vectors[2] = .{ 0.0, 0.1 };
    vectors[3] = .{ -0.1, 0.0 };
    vectors[4] = .{ 0.0, -0.1 };

    // Cluster 2: around (10, 10)
    vectors[5] = .{ 10.0, 10.0 };
    vectors[6] = .{ 10.1, 10.0 };
    vectors[7] = .{ 10.0, 10.1 };
    vectors[8] = .{ 9.9, 10.0 };
    vectors[9] = .{ 10.0, 9.9 };

    var kmeans = KM.init(allocator, 2, 10, 1e-4);
    defer kmeans.deinit();

    try kmeans.train(vectors);

    try std.testing.expect(kmeans.centroids != null);
    try std.testing.expectEqual(2, kmeans.centroids.?.len);

    const c1 = kmeans.centroids.?[0];
    const c2 = kmeans.centroids.?[1];

    const d1 = @reduce(.Add, c1 * c1);
    const d2 = @reduce(.Add, c2 * c2);

    const has_low = (d1 < 1.0) or (d2 < 1.0);
    const has_high = (d1 > 190.0) or (d2 > 190.0);

    try std.testing.expect(has_low);
    try std.testing.expect(has_high);
}

// ┌─────────────────────────────────────────────────────────────┐
// │  Benchmark: IVF vs Brute-Force Search                       │
// │                                                             │
// │  Measures wall-clock time for:                              │
// │    1. Brute-force: scan all N vectors                       │
// │    2. IVF: scan nprobe clusters (~N/nlist vectors each)     │
// │                                                             │
// │  Expected speedup ≈ nlist / nprobe (in ideal conditions)    │
// │                                                             │
// │  Example with N=10000, nlist=100, nprobe=10:                │
// │    Brute-force: 10000 distance computations                 │
// │    IVF: 100 (centroids) + 10×100 (vectors) = 1100           │
// │    Theoretical speedup: ~9×                                 │
// └─────────────────────────────────────────────────────────────┘
test "benchmark: IVF vs brute-force speedup" {
    const allocator = std.testing.allocator;
    const IVF = IVFIndex(64); // 64-dim vectors (realistic embedding size)

    // Dataset parameters
    const n_vectors: usize = 10_000;
    const n_clusters: usize = 4;
    const nlist: usize = 100; // 100 clusters → ~100 vectors per cluster
    const n_queries: usize = 100;
    const k: usize = 10;

    // Generate synthetic clustered dataset
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    const vectors = try allocator.alloc(IVF.Vec, n_vectors);
    defer allocator.free(vectors);

    // Create cluster centers
    var centers: [n_clusters]IVF.Vec = undefined;
    for (0..n_clusters) |c| {
        var center: IVF.Vec = undefined;
        for (0..64) |d| {
            center[d] = @as(f32, @floatFromInt(c * 10)) + random.float(f32);
        }
        centers[c] = center;
    }

    // Generate vectors around cluster centers
    for (0..n_vectors) |i| {
        const center = centers[i % n_clusters];
        var vec: IVF.Vec = undefined;
        for (0..64) |d| {
            vec[d] = center[d] + (random.float(f32) - 0.5) * 2.0;
        }
        vectors[i] = vec;
    }

    // Generate random queries
    const queries = try allocator.alloc(IVF.Vec, n_queries);
    defer allocator.free(queries);
    for (0..n_queries) |q| {
        var query: IVF.Vec = undefined;
        for (0..64) |d| {
            query[d] = random.float(f32) * 40.0; // spread across cluster space
        }
        queries[q] = query;
    }

    // Build IVF index
    var index = try IVF.build(allocator, vectors, nlist, 50, 1e-4);
    defer index.deinit();

    // Benchmark brute-force
    var brute_force_timer = try std.time.Timer.start();
    for (queries) |query| {
        // Brute-force: compute distance to all vectors
        var best_dist: f32 = std.math.floatMax(f32);
        for (vectors) |vec| {
            const diff = query - vec;
            const dist = @reduce(.Add, diff * diff);
            best_dist = @min(best_dist, dist);
        }
        std.mem.doNotOptimizeAway(best_dist);
    }
    const brute_force_ns = brute_force_timer.read();

    // Benchmark IVF at different nprobe values
    const nprobe_values = [_]usize{ 1, 5, 10, 20 };
    var ivf_times: [nprobe_values.len]u64 = undefined;

    for (nprobe_values, 0..) |nprobe, idx| {
        var ivf_timer = try std.time.Timer.start();
        for (queries) |query| {
            const results = try index.search(query, k, nprobe);
            defer allocator.free(results);
            std.mem.doNotOptimizeAway(results.ptr);
        }
        ivf_times[idx] = ivf_timer.read();
    }

    // Print results
    std.debug.print("\n", .{});
    std.debug.print("┌─────────────────────────────────────────────────────────┐\n", .{});
    std.debug.print("│  IVF Benchmark Results                                  │\n", .{});
    std.debug.print("│  N={d}, nlist={d}, k={d}, queries={d}             │\n", .{ n_vectors, nlist, k, n_queries });
    std.debug.print("├─────────────────────────────────────────────────────────┤\n", .{});
    std.debug.print("│  Brute-force: {d:.2} ms                                 │\n", .{@as(f64, @floatFromInt(brute_force_ns)) / 1_000_000.0});
    std.debug.print("├─────────────────────────────────────────────────────────┤\n", .{});

    for (nprobe_values, 0..) |nprobe, idx| {
        const ivf_ms = @as(f64, @floatFromInt(ivf_times[idx])) / 1_000_000.0;
        const speedup = @as(f64, @floatFromInt(brute_force_ns)) / @as(f64, @floatFromInt(ivf_times[idx]));
        std.debug.print("│  IVF nprobe={d:2}: {d:.2} ms  (speedup: {d:.1}×)         │\n", .{ nprobe, ivf_ms, speedup });
    }

    std.debug.print("└─────────────────────────────────────────────────────────┘\n", .{});

    // Verify IVF is faster than brute-force at low nprobe
    try std.testing.expect(ivf_times[0] < brute_force_ns);
}
