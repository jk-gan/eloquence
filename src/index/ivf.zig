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
                    var min_distance_squared: f32 = distance_squared(vec, centroids[0]);
                    for (centroids[0..i]) |centroid| {
                        const d = distance_squared(vec, centroid);
                        min_distance_squared = math.min(min_distance_squared, d);
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
